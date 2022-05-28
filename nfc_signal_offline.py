import logging
import textwrap
from matplotlib import figure

import numpy as np
import pandas as pd
from numpy.random import normal
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from colorama import Style, Fore

from sklearn.cluster import KMeans

from nfc_signal_help import *
from device import *
from dataclasses import dataclass


# conversion to real time
# 1. find where em is active using moving average
# 2. find start of message using moving average: there is a change in the moving average
# 3. find end of message
# 4. demodulate found message
# 5. fine end of em

@dataclass
class NfcSignal:
    # data
    signal: np.ndarray
    expected_file: str = None
    libnfc_file: str = None
    output_file: str = None
    # demodulation
    demodulate_device: Device = None
        # -1 = no mean,
        # 1 = Take mean of message and multiple noise representations
        # 0 = Take mean of multiple similar samples
    attack_mode: int = 0
    mean_samples: int = 0
    window_size: int = 128
    window_size_avg: int = 4096
    fixed_message_size: int = 14000
    # plotting/showing data
    show_plots: bool = False
    hide_demodulated_data: bool = True
    Fs: int = 1695000 # 1695000, 2542500
    message_batch_size: int = 0
    Fc: int = 13.56e6

    def __post_init__(self):
        self.__create_logger()
        # for stats
        self.__init_set_stats_params()

        # for demodulator
        self.__init_set_demodulator_params()
        self.__init_set_thresholds()
        self.logger.debug("em_threshold %f, card_threshold %f, reader_threshold %f, max_in_em %f" % (self.em_threshold, self.card_threshold, self.reader_threshold, self.max_in_em))

        # output of find_em/find_meassages/demodulator
        self.__init_output_vars()

        # start deteciton for signal
        self.start_em, self.end_em = self.__find_em(
            self.moving_avg, 
            self.em_threshold, 
            self.window_size_avg
        ) # TODO: how to debug this

        self.logger.debug("self.start_em index = %s" % self.start_em)
        self.logger.debug("self.end_em index = %s" % self.end_em)

        # find messages start and end
        self.message_start, self.message_end, self.gradient, self.gradient_diff = self.__find_messages(
            self.start_em, 
            self.end_em, 
            self.signal_normalized, 
            self.reader_threshold, 
            self.card_threshold, 
            self.attack_mode, 
            self.window_size,
            self.fixed_message_size
        ) # TODO: how to debug this

        self.logger.debug("self.message_start length: %s" % [len(x) for x in self.message_start])
        self.logger.debug("self.message_end length: %s" % [len(x) for x in self.message_end])

        # detect messages and get their type
        self.message_type, self.message_detected, self.last_max_reader, self.signal_without_reader = self.__get_messages_type(
            self.message_start, 
            self.message_end, 
            self.signal_normalized, 
            self.signal, 
            self.gradient
        )

        # how many reader messages detected? Device.READER, True over all Device.READER
        reader_detected_count = np.sum(np.array(self.message_detected)[np.array(self.message_type) == Device.READER])
        # how many tag messages detected? Device.TAG, True over all Device.TAG
        tag_detected_count = np.sum(np.array(self.message_detected)[np.array(self.message_type) == Device.TAG])
        self.logger.debug("reader messages detected: %s" % reader_detected_count)
        self.logger.debug("tag messages detected (subsection of messages which are demodulated): %s" % tag_detected_count)

    def __init_set_stats_params(self):
        self.message_expected = get_libnfc_sequence(self.expected_file) # expected hex representation
        message_libnfc = get_libnfc_sequence(self.libnfc_file)
        self.libnfc_correct = np.array(message_libnfc) == np.array(self.message_expected)
        
        D_b = self.Fc/self.window_size # bit rate
        self.ETU = 1/D_b # elementary time unit
        self.N_samples_per_symbol = int(Fs/D_b) # number of samples per symbol: 16, 24
        

    def __init_set_demodulator_params(self):
        # heavy on memory: moving_avg, gradient, gradient_diff, signal_without_reader,signal_normalized
        self.moving_avg = moving_average(self.signal, self.window_size_avg) # for em detection
        self.max_in_em = np.max(self.moving_avg)
        self.em_threshold = self.max_in_em * 1/2 # to check if em active
        self.gradient = np.array([]) # gradient: used to split messages and distinguish between card and reader
        self.gradient_diff = np.array([]) # gradient_diff only the last and and penultimate are necessary for calculating when messages start and end
        self.signal_without_reader = np.array([])
        self.max_in_normalized_message = 1
        # TODO: review, might introduce demodulation errors
        signal_rectified = self.max_in_normalized_message + self.signal[:-(self.window_size-1)] - np.max(sliding_window_view(self.signal, window_shape = self.window_size), axis = 1)
        self.signal_normalized = (signal_rectified - np.min(signal_rectified))/(1 - np.min(signal_rectified))
        signal_rectified = None

    def __init_set_thresholds(self):
        self.card_threshold = self.max_in_normalized_message * 3/4 # to check if card
        # self.card_threshold = self.max_in_normalized_message * 9/10 # to check if card
        self.reader_threshold = self.max_in_normalized_message * 1/2 # to check if reader
        

    def __init_output_vars(self):
        # find_em
        self.start_em = [] #em field start
        self.end_em = [] #em field end
        # find_messsages
        self.message_start = [] # used to say when messages start
        self.message_end = [] # used to say when messages end
        self.last_max_reader = [] # last peak of the reader transmissions
        self.message_detected = [] # detection of a message
        self.message_type = [] # used to distinguish between reader and card
        self.message_hex = [] # hex representation of message
        self.message_correct = [] # was demodulation correct?

    def __create_logger(self):
        # create logger
        self.logger = logging.getLogger('simple_example')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter
        formatter = logging.Formatter('(%(levelname)s) %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def save_hex(self):
        if self.output_file != None:
            np.savetxt(self.output_file, self.message_hex, fmt="%s", delimiter='\n')

    '''
    Plot specgram
    '''
    def specgram(self, NFFT=256, no_reader=False):
        if no_reader:
            return specgram(self.signal_without_reader, Fs=self.Fs, NFFT=NFFT, ETU=self.ETU, N_samples_per_symbol=self.N_samples_per_symbol)
        else:
            return specgram(self.signal, Fs=self.Fs, NFFT=NFFT, ETU=self.ETU, N_samples_per_symbol=self.N_samples_per_symbol)

    def get_animated_psd(self, Fs=1.8e6,NFFT=None,Fc=0):
        return get_animated_psd(self.signal, Fs, NFFT, Fc)

    def specgram_mean(self, spectrum, NFFT=256):
        return specgram_mean(spectrum, self.message_start, self.message_type, NFFT)

    def noise_peak_frequencies(spectrum_mov_avg, th_var=4):
        return noise_peak_frequencies(spectrum_mov_avg=spectrum_mov_avg, th_var=th_var)

    def plot_clean_signal_max_window(self):
        # clean signal and max window
        max_window = np.max(sliding_window_view(self.signal, window_shape = 4096), axis = 1)
        print(self.signal.shape[0], self.Fs)
        time_axis = np.arange(0,self.signal.shape[0]*1/self.Fs,1/self.Fs)
        plt.figure(figsize=(20,8))
        plt.title("clean signal and max window")
        plt.plot(time_axis[:max_window.size], max_window, 'b', label="max in window")
        plt.plot(time_axis, self.signal, label="signal")
        plt.plot(self.start_em[0]*1/self.Fs, self.em_threshold, 'x') # start
        plt.plot(self.end_em[0]*1/self.Fs, self.em_threshold, '.') # stop
        for a,b in zip(self.message_start, self.message_end):
            for c,d in zip(a,b):
                plt.plot(c*1/self.Fs, self.em_threshold, 'x')
                plt.plot(d*1/self.Fs, self.em_threshold, '*')
        plt.plot(time_axis, self.moving_avg, label="moving average")
        plt.xlabel('time [s]')
        plt.ylabel('peak voltage [V]')
        plt.legend()
        plt.show()

    def plot_normalized(self):
        # normalized
        plt.figure(figsize=(20,8))
        plt.title("signal normalized")
        time_axis = np.arange(0,self.signal_normalized.shape[0]*1/self.Fs,1/self.Fs)
        plt.plot(time_axis, self.signal_normalized)
        plt.xlabel('time [s]')
        plt.ylabel('normalized peak voltage [V]')
        plt.show()

    def plot_gradient(self):
        # gradient and gradient_diff
        plt.figure(figsize=(20,8))
        plt.title("gradient and gradient_diff")
        time_axis = np.arange(0,self.gradient.shape[0]*1/self.Fs,1/self.Fs)
        plt.plot(time_axis, self.gradient)
        plt.plot(time_axis, self.gradient_diff)
        plt.xlabel('time [s]')
        plt.ylabel('gradient amplitude')
        plt.show()

    def plot_stats(self):
        time_axis = np.arange(0,self.message_correct.shape[0]*1/self.Fs,1/self.Fs)
        plt.plot(time_axis, self.message_correct)
        plt.show()

    ''' 
    Reports demodulation stats
    '''
    def demodulation_stats(self):
        def print_demodulation_info(df):
            self.logger.info("RTL-SDR")
            self.logger.info("\tcorrect reader demodulations over detected reader messages: %i/%i = %f" % (df['rtl_reader_correct'][0], df['rtl_reader_tot'][0], df['rtl_reader_correct'][0] / df['rtl_reader_tot'][0]))
            self.logger.info("\tcorrect tag demodulations over detected tag messages: %i/%i = %f" % (df['rtl_tag_correct'][0], df['rtl_tag_detected'][0], df['rtl_tag_correct'][0]/df['rtl_tag_detected'][0]))
            self.logger.info("\tcorrect tag demodulations over total tag replies without blck: %i/%i = %f" % (df['rtl_tag_correct'][0], df['tag_expected_tot'][0], df['rtl_tag_correct'][0]/df['tag_expected_tot'][0]))
            self.logger.info("LIBNFC")
            self.logger.info("\tcorrect tag demodulations over total tag replies without blck: %i/%i = %f" % (df['libnfc_tag_correct'][0], df['tag_expected_tot'][0], df['libnfc_tag_correct'][0]/df['tag_expected_tot'][0]))
            self.logger.info("RTL-SDR and LIBNFC comparison (tag only)")
            self.logger.info("\tCorrect messages in rtlsdr and wrong in libnfc %i (are we doing better than libnfc?)" % np.sum(df['correct_rtlsdr_wrong_libnfc'][0]))
            self.logger.info("\tCorrect messages in libnfc and wrong in rtlsdr %i (are we doing worse than libnfc?)" % np.sum(df['correct_libnfc_wrong_rtlsdr'][0]))
            self.logger.info("\tCorrect messages in libnfc or rtlsdr %i (how are we doing overall?)" % np.sum(df['correct_libnfc_rtlsdr'][0]))
      
        # reader and tag indexes
        reader_indexes = np.array(self.message_type) == Device.READER
        # calculate the first None in message_expected
        wrong_message_index = np.where(np.array(self.message_expected)==None)[0][0] + 1
        # exclude wrong_message_indexes
        # and the first five samples
        tag_indexes = np.logical_and(
            np.array(self.message_type) == Device.TAG, 
            np.arange(0,len(self.message_type),1) % wrong_message_index != wrong_message_index - 1)
        # TODO: remove the first mean_samples
        # TODO: check how you take the mean
        # first_mean_samples = np.where(tag_indexes == True)[0][:self.mean_samples*self.message_batch_size]
        # self.logger.debug("Removing first_mean_samples from tag_indexes: %s" %first_mean_samples)
        # for m in first_mean_samples:
        #     tag_indexes[m] = False

        # how many reader messages detected? Device.READER, True over all Device.READER
        detected_reader_messages = np.sum(np.array(self.message_detected)[reader_indexes])
        # how many tag messages detected? Device.TAG, True over all Device.TAG
        detected_tag_messages = np.sum(np.array(self.message_detected)[tag_indexes])

        # how many correct reader messages?
        correct_reader_demodulation = np.sum(np.array(self.message_correct)[reader_indexes])
        # how many correct tag messages?
        correct_tag_demodulation = np.sum(np.array(self.message_correct)[tag_indexes])

        # remove empty tag messages, those where you have None
        empty_tag_messages_count = np.sum(np.array(self.message_expected) == None)
        detected_tag_no_empty_messages = detected_reader_messages - empty_tag_messages_count

        row = [
            correct_reader_demodulation, 
            detected_reader_messages, 
            correct_tag_demodulation, 
            detected_tag_messages, 
            detected_tag_no_empty_messages
        ]

        if len(self.libnfc_correct) == tag_indexes.size:
            libnfc_tag_correct = np.sum(np.array(self.libnfc_correct)[tag_indexes])

            rtlsdr_correct = np.array(self.message_correct)
            correct_rtlsdr_wrong_libnfc = np.logical_and(rtlsdr_correct[tag_indexes], np.logical_not(self.libnfc_correct[tag_indexes]))# correct messages in rtl-sdr, wrong in libnfc (useful cases to send to libnfc)
            correct_libnfc_wrong_rtlsdr = np.logical_and(np.logical_not(rtlsdr_correct[tag_indexes]), self.libnfc_correct[tag_indexes])# correct in libnfc, wrong in rtl-sdr (bad demodulation from my side)
            correct_libnfc_rtlsdr = np.logical_or(rtlsdr_correct[tag_indexes], self.libnfc_correct[tag_indexes])# correct in both

            row.append(libnfc_tag_correct)
            row.append(np.sum(correct_rtlsdr_wrong_libnfc))
            row.append(np.sum(correct_libnfc_wrong_rtlsdr))
            row.append(np.sum(correct_libnfc_rtlsdr))
        else:
            for i in range(4):
                row.append(None)
        row.append(self.mean_samples)
        # row
        df = pd.DataFrame([row], columns = 
            ['rtl_reader_correct', 'rtl_reader_tot', 'rtl_tag_correct', 'rtl_tag_detected', 'tag_expected_tot',
            'libnfc_tag_correct', 'correct_rtlsdr_wrong_libnfc', 
            'correct_libnfc_wrong_rtlsdr', 'correct_libnfc_rtlsdr', 'means']
        )
        print_demodulation_info(df)

        return df

    '''
    Find EM field of the NFC reader.
    Input: self.moving_avg, self.em_threshold
    Output: self.start_em (array of start points for EM), 
            self.end_em (array of end points for EM), 
    '''
    @staticmethod
    def __find_em(moving_avg, em_threshold, window_size_avg):
        # find where the moving average if greater than the EM threshold
        moving_avg_ones = np.where(moving_avg > em_threshold)[0]
        # split moving_avg_ones into multiple sub arrays
        em_indexes = np.split(moving_avg_ones, np.where(np.diff(moving_avg_ones) != 1)[0])
        # store only start and end of em_indexes
        start_em = []
        end_em = []
        for em_index in em_indexes:
            start_em.append(em_index[0] + int(window_size_avg/2))
            end_em.append(em_index[-1] - int(window_size_avg/2))
        return start_em, end_em

    # must be run after __find_em
    def __get_em_active_index(self):
        return self.start_em[0]

    # must be run after __find_em
    def get_em_power_location(self, NFFT=256):
        data = self.signal[self.__get_em_active_index() : self.__get_em_active_index()+self.fixed_message_size]
        spectrum, freqs, t = mlab.specgram(data, Fs=self.Fs, NFFT=NFFT, sides='twosided')
        # see Welch's average periodogram
        # https://ccrma.stanford.edu/~jos/sasp/Welch_s_Method.html#:~:text=Welch's%20method%20%5B297%5D%20(also,for%20each%20block%2C%20and%20averaging.&text=is%20the%20rectangular%20window%2C%20the,overlapping%20successive%20blocks%20of%20data.
        spectrum = 10*np.log10(np.flip(spectrum.transpose(),axis=0)) # spectrum correction
        return np.mean(spectrum)

    # must be run after __find_em
    def get_em_mean(self):
        data = self.signal[self.__get_em_active_index() : self.__get_em_active_index()+self.fixed_message_size]
        return np.mean(data)

    # must be run after __find_em
    def get_em_var(self):
        data = self.signal[self.__get_em_active_index() : self.__get_em_active_index()+self.fixed_message_size]
        return np.var(data)

    # must be run after __find_em
    def get_em_std(self):
        data = self.signal[self.__get_em_active_index() : self.__get_em_active_index()+self.fixed_message_size]
        return np.std(data)


    @staticmethod
    def __get_messages_type(
        message_start_list, 
        message_end_list, 
        signal_normalized, 
        signal, 
        gradient, 
        small_offset=400):
        def __detect_and_get_type(gradient, m_s, m_e, possible_tag, noise):
            if gradient[int((m_s+m_e)/2)] == 1:
                return Device.READER, True
            elif gradient[int((m_s+m_e)/2)] == 1/2 or np.abs(np.std(possible_tag)-np.std(noise)) > 0.01:
                return Device.TAG, True
            else:
                return Device.TAG, False
            # TODO: add rare case, reader not detected/present
        
        message_type = []
        message_detected = []
        last_max_reader = []
        signal_without_reader = signal.copy()
        # now iterate and detect messages alternating between reader and tag
        for message_start, message_end in zip(message_start_list, message_end_list):
            for m_s,m_e in zip(message_start, message_end):

                possible_tag = signal_normalized[m_s:m_s+small_offset]
                noise = signal_normalized[m_e-small_offset:m_e]

                m_t, m_d = __detect_and_get_type(gradient, m_s, m_e, possible_tag, noise)
                message_type.append(m_t)
                message_detected.append(m_d)

                if m_t == Device.READER and m_d:
                    last_max_reader.append(m_e)
                    # signal_without_reader[m_s:m_e] = signal[m_s - (m_e - m_s) - 100 : m_s - 100] # zero out message in those portions
                    signal_without_reader[m_s:m_e] = signal[m_s - (m_e - m_s) : m_s] # zero out message in those portions

        return message_type, message_detected, last_max_reader, signal_without_reader

    @staticmethod
    def __find_messages(
        start_em, 
        end_em, 
        signal_normalized, 
        reader_threshold, 
        card_threshold, 
        attack_mode, 
        window_size,
        fixed_message_size):
        def __get_bits(em_trim, reader_threshold, card_threshold, attack_mode):
            em_bits = np.zeros(em_trim.size)

            # boolean array to say if the indexes are part of reader
            is_reader = em_trim <= reader_threshold
            # sets em_bits to 1 if the indexes are part of reader
            em_bits[is_reader] = 1
            # if not attacking blocking cards try detecting the card
            if attack_mode == -1:
                # boolean array to say if the indexes are part of a card
                is_card = np.logical_and(em_trim > reader_threshold, em_trim <= card_threshold)
                # sets em_bits to 1/2 if the indexes are part of card
                em_bits[is_card] = 1/2
            return em_bits

        message_start_list = []
        message_end_list = []
        gradient = np.zeros(signal_normalized.size)
        gradient_diff = np.zeros(signal_normalized.size)

        # cycle through the EM field indexes
        for start, end in zip(start_em, end_em):
            # get signal portion
            em_trim = signal_normalized[start:end]
            # bit representation of the trimmed signal
            em_bits = __get_bits(em_trim, reader_threshold, card_threshold, attack_mode)

            # get sliding window maximum
            max_window = np.max(sliding_window_view(em_bits, window_shape = window_size), axis = 1)
            # set gradient and gradient_diff
            gradient[start:start+max_window.size] = max_window
            gradient_diff[start:start+max_window.size-1] = np.diff(gradient[start:start+max_window.size])
            
            # find message_start
            message_start = np.where(gradient_diff > 0)[0]
            # remove duplicates
            # message_start = message_start[np.diff(np.append(message_start,np.inf)) > self.window_size]
            # find message_end
            message_end = np.where(gradient_diff < 0)[0]
            # remove duplicates
            # message_end = message_end[np.diff(np.append(message_end,np.inf)) > self.window_size]
            # add self.window_size to avoid detection errors
            message_end += window_size

            # add potential tag messages when attacking
            if attack_mode != -1:
                message_start = np.insert(message_start, np.arange(1,message_end.size+1), message_end + 1)
                message_end = np.insert(message_end, np.arange(1,message_end.size+1), message_end + fixed_message_size + 1)

            message_start_list.append(message_start)
            message_end_list.append(message_end)

        return message_start_list, message_end_list, gradient, gradient_diff
        
    def start_demodulation(self):
        # call private start_demodulation
        self.message_hex = self.__start_demodulation(
            self.message_start, 
            self.message_end, 
            self.message_detected, 
            self.message_type, 
            self.signal_normalized, 
            self.attack_mode, 
            self.mean_samples, 
            self.message_batch_size, 
            self.demodulate_device, 
            self.show_plots, 
            self.N_samples_per_symbol, 
            self.hide_demodulated_data)

        # initialize message_correct for statistics
        if self.message_expected != None:
            self.message_correct = np.array(self.message_hex) == np.array(self.message_expected)

    @staticmethod
    def __attack_message(message, signal_normalized, attack_mode, m_t, m_d, mean_samples, message_count, message_batch_size, message_start, message_end, message_type):
        if attack_mode == 1 and mean_samples > 0 and m_t == Device.TAG:
            pass
        #     # Take mean of message and multiple noise representations
        #     mean_tag_message = [message]
        #     # self.noise_mean, self.noise_var
        #     for _ in range(mean_samples):
        #         mean_tag_message.append(normal(loc=self.get_em_mean(), scale=self.get_em_std(), size=message.size))
        #     message = np.mean(mean_tag_message, axis=0) # replace message with mean
        #     mean_tag_message = None

        # Take mean of multiple similar samples
        # There is a batch of e.g. 4x80 messages
        # Replace message with mean
        elif attack_mode == 0 and mean_samples > 0 and m_t == Device.TAG:
            # previously detected messages
            # (message_type == TAG and message_detected) and (np.arange(message_count) % self.message_batch_size) == ((message_count) % self.message_batch_size))
            other_indexes = np.where(np.logical_and(np.logical_and(np.array(message_type[:message_count]) == Device.TAG, m_d), (np.arange(message_count) % message_batch_size) == ((message_count) % message_batch_size)))[0] # known tag indexes
            
            # self.logger.info("%i, mean of %s", message_count, other_indexes[-self.mean_samples:])
            mean_tag_message = [message]
            # take mean of self.mean_samples samples
            for x in other_indexes[-mean_samples:]:
                other_message_start = message_start[int(x)]
                other_message_end = message_end[int(x)]
                other_tag_message = signal_normalized[other_message_start:other_message_end]
                mean_tag_message.append(other_tag_message)
                other_tag_message = None
            message = np.mean(mean_tag_message, axis=0) # replace message with mean
            mean_tag_message = None
        return message

    '''
    Loops through messages and demodulates
    '''
    @staticmethod
    def __start_demodulation(
        message_start_list, 
        message_end_list, 
        message_detected, 
        message_type, 
        signal_normalized, 
        attack_mode, 
        mean_samples, 
        message_batch_size, 
        demodulate_device, 
        show_plots, 
        N_samples_per_symbol, 
        hide_demodulated_data, 
    ):
        message_hex = []
        message_count = 1 if demodulate_device == Device.TAG else 0
        
        
        for message_start, message_end in zip(message_start_list, message_end_list):
            for m_s, m_e, m_d, m_t in zip(message_start, message_end, message_detected, message_type):
                if m_d:
                    message = signal_normalized[m_s:m_e]

                    if attack_mode > -1:
                        message = NfcSignal.__attack_message(
                            message, 
                            signal_normalized,
                            attack_mode, 
                            m_t, 
                            m_d,
                            mean_samples, 
                            message_count, 
                            message_batch_size, 
                            message_start, 
                            message_end,
                            message_type)

                    # can be reader or tag
                    hex_decoded = NfcSignal.perform_demodulation(
                        message,
                        message_count=message_count, 
                        demodulate_device=demodulate_device,
                        device_flag=m_t,
                        show_plots=show_plots,
                        from_start=m_s,
                        to_end=m_e,
                        N_samples_per_symbol=N_samples_per_symbol,
                        hide_demodulated_data=hide_demodulated_data,
                    )
                    
                    hex_decoded = hex_decoded.rstrip()

                    # save results of decoding
                    message_hex.append(hex_decoded)
                else:
                    message_hex.append(None)

                message_count += 1
        return message_hex

        
    '''
    Performs demodulation given the message slice
    '''
    @staticmethod
    def perform_demodulation(message, device_flag, message_count=0, demodulate_device=None, show_plots=False, from_start=0, to_end=0, N_samples_per_symbol=16, hide_demodulated_data=False, k_means_offset=1000):
        # print("Detected: %s, Demodulating only: %s" % (device_flag, demodulate_device))
        def __get_device_tag(device_flag, demodulate_device):
            if device_flag == Device.READER and (demodulate_device == Device.READER or demodulate_device == None):
                return Device.READER
            elif device_flag == Device.TAG and (demodulate_device == Device.TAG or demodulate_device == None):
                return Device.TAG 
            return None

        def __get_bits(message, separation_value):
            bit_signal = np.zeros(message.size)
            bit_signal[message < separation_value] = 0
            bit_signal[message >= separation_value] = 1
            return bit_signal[np.argmin(bit_signal):] # sync operation

        def demodulate_tag(bit_signal, N_samples_per_symbol):
            demodulated_sequence = ''
            for i in range(0, bit_signal.size - N_samples_per_symbol, N_samples_per_symbol):
                # pattern strategy
                pattern_to_match = bit_signal[i:i+N_samples_per_symbol]
                h = int(N_samples_per_symbol/2) # 8,12
                sequence_D = [0]*h + [1]*h # 01 = 0x1
                sequence_E = [1]*h + [0]*h # 10 = 0x2
                sequence_F = [1]*h + [1]*h # 11 = 0x3
                matched_pattern = pattern_match(pattern_to_match, sequence_D, sequence_E, sequence_F)
                if matched_pattern == sequence_D and len(demodulated_sequence) == 0:
                    demodulated_sequence += "S"
                # S letter check to avoid demodulation of wrong messages
                elif matched_pattern == sequence_D and len(demodulated_sequence) > 0 and demodulated_sequence[0] == "S":
                    demodulated_sequence += "1"
                elif matched_pattern == sequence_E and len(demodulated_sequence) > 0 and demodulated_sequence[0] == "S":
                    demodulated_sequence += "0"
                elif matched_pattern == sequence_F and len(demodulated_sequence) > 0 and demodulated_sequence[0] == "S":
                    demodulated_sequence += "E"
                    break
            return demodulated_sequence

        def demodulate_reader(bit_signal, N_samples_per_symbol):
            demodulated_sequence = ""
            last_bit = ""
            for i in range(0,bit_signal.size - N_samples_per_symbol, N_samples_per_symbol):
                # pattern matching
                pattern_to_match = bit_signal[i:i+N_samples_per_symbol]
                q = int(N_samples_per_symbol/4) # 4,6
                sequence_X = [1]*q + [1]*q + [0]*q + [1]*q # 1101 = 0xd
                sequence_Y = [1]*q + [1]*q + [1]*q + [1]*q # 1111 = 0xf
                sequence_Z = [0]*q + [1]*q + [1]*q + [1]*q # 0111 = 0x7

                matched_pattern = pattern_match(bit_signal[i:i+N_samples_per_symbol], sequence_X, sequence_Y, sequence_Z)
                if len(demodulated_sequence) > 0:
                    last_bit = demodulated_sequence[-1]
                else:
                    last_bit = ""
                # logic 1: sequence X
                if matched_pattern == sequence_X:
                    demodulated_sequence += "1"
                # end bit E: logic0 | Y
                elif last_bit == "0" and matched_pattern == sequence_Y:
                    demodulated_sequence = demodulated_sequence[:-1] + "E" # replace last zero with E
                    break
                # logic 0: sequence Y or Z from second logic 0 or if there was a start bit S
                elif matched_pattern == sequence_Y or (last_bit == "0" and matched_pattern == sequence_Z) or (last_bit == "S" and matched_pattern == sequence_Z):
                    demodulated_sequence += "0"
                # start bit S: sequence Z
                elif matched_pattern == sequence_Z:
                    demodulated_sequence += "S"
            return demodulated_sequence
            
        def __cut_start_end_bit(demodulated_sequence):
            # cut start and end bit
            if demodulated_sequence[0] == "S":
                bits_only = demodulated_sequence[1:] # cut start bit
            else:
                bits_only = demodulated_sequence
            if demodulated_sequence[-1] == "E":
                bits_only = bits_only[:-1] # cut end bit
            return bits_only

        def __print_bytes(hex_decoded, bits_only, message_count, device_flag):
            pcks = textwrap.wrap(bits_only, 9)
            print_arrow = "<=" if device_flag == Device.TAG else "=>"
            print_info = f"(%i) {print_arrow} {Fore.GREEN}" % message_count
            for pck in pcks:
                try:
                    hex_decoded += "%02x " % int(pck[:-1][::-1], 2)
                except:
                    hex_decoded += "err"
            print_info += hex_decoded
            return hex_decoded, pcks, print_info
        
        def __print_parity(pcks, print_info):
            print_info += f"{Style.RESET_ALL}- PA: "
            for pck in pcks:
                # 1 if the number of 1s is odd in pck
                parity_check = (pck[:-1].count("1") % 2 == 0) == (pck[-1] == "1")
                parity_ok = "OK " if parity_check else "ER "
                print_info += parity_ok
            return print_info

        def __print_bcc(pcks, print_info):
            try:
                if (int(pcks[0][:-1][::-1],2) == 0x93 and int(pcks[1][:-1][::-1],2) == 0x70) or (int(pcks[0][:-1][::-1],2) == 0x95 and int(pcks[1][:-1][::-1],2) == 0x70):
                    bcc_check = (int(pcks[2][:-1],2) ^ int(pcks[3][:-1],2) ^ int(pcks[4][:-1],2) ^ int(pcks[5][:-1],2)) == (int(pcks[6][:-1],2) + int(pcks[7][:-1],2))
                    print_info += f"{Style.RESET_ALL}- BCC: %d " % ((int(pcks[2][:-1],2) ^ int(pcks[3][:-1],2) ^ int(pcks[4][:-1],2) ^ int(pcks[5][:-1],2)))
            except:
                print_info += "- BCC: err"
            return print_info

        def get_demodulated_sequence_tag(demodulated_sequence, message_count, hide_demodulated_data):
            hex_decoded = ''
            if len(demodulated_sequence) > 2: # PICC frame
                bits_only = __cut_start_end_bit(demodulated_sequence)
                hex_decoded, pcks, print_info = __print_bytes(hex_decoded, bits_only, message_count, device_flag)
                print_info = __print_parity(pcks, print_info)
                print_info += ", bits: %s" % (demodulated_sequence)
                # TODO: how to calculate CRC written in ANNEX B of 14443-3
                # crc check on all data sent
                if not hide_demodulated_data:
                    print(print_info)
            return hex_decoded

        def get_demodulated_sequence_reader(demodulated_sequence,message_count,hide_demodulated_data):
            hex_decoded = ''
            # short frame       s | b1 b2 b3 b4 b5 b6 b7 | e
            # standard frame    s | b1 b2 b3 b4 b5 b6 b7 b8 | p | b1 b2 b3 b4 b5 b6 b7 b8 | p | e
            if len(demodulated_sequence) == 9: # short frame
                bits_only = __cut_start_end_bit(demodulated_sequence)
                pck = bits_only
                print_info = f"(%i) => {Fore.GREEN}" % message_count
                hex_decoded += "%02x " % int(pck[::-1], 2) # reverse bits order
                print_info += hex_decoded
                print_info += f"{Style.RESET_ALL}, bits: %s" % (demodulated_sequence)
                if not hide_demodulated_data:
                    print(print_info)
            elif len(demodulated_sequence) > 9: # standard frame
                bits_only = __cut_start_end_bit(demodulated_sequence)
                hex_decoded, pcks, print_info = __print_bytes(hex_decoded, bits_only, message_count, device_flag)
                # bcc check on 0x93 0x70 and 0x95 0x70: xor from 2:6
                print_info = __print_bcc(pcks, print_info)
                print_info = __print_parity(pcks, print_info)
                print_info += ", bits: %s" % (demodulated_sequence)
                # TODO: how to calculate CRC written in ANNEX B of 14443-3
                # crc check on 0x30 0x04, 0x50 0x00
                if not hide_demodulated_data:
                    print(print_info)
            return hex_decoded

        device_tag = __get_device_tag(device_flag, demodulate_device)

        hex_decoded = ""
        if device_tag == None:
            return hex_decoded
            
        # kmeans mean for semparation over first 1000 samples to avoid errors when taking large messages
        kmeans = KMeans(n_clusters=2).fit(message[:k_means_offset].reshape(-1,1))
        cluster_centers_ = kmeans.cluster_centers_
        separation_value = np.mean(cluster_centers_)

        # set bits to 0 and 1 according to the mean of the centroids
        bit_signal = __get_bits(message, separation_value)

        # plots
        if show_plots:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,4))
            ax1.set_title("bit signal from %i to %i" % (from_start, to_end))
            ax1.plot(bit_signal, '.') # another plot of this
            ax1.plot(message, 'x')
            ax1.plot(message)
            ax1.plot(separation_value*np.ones(message.size), '--', label="separation_value")
            ax1.legend()

            ax2.set_title("bits only from %i to %i" % (from_start, to_end))
            ax2.plot(bit_signal)
            plt.show()

        # text demodulation
        if device_tag == Device.TAG:
            demodulated_sequence = demodulate_tag(bit_signal, N_samples_per_symbol)
            hex_decoded = get_demodulated_sequence_tag(demodulated_sequence, message_count, hide_demodulated_data)
        elif device_tag == Device.READER:
            demodulated_sequence = demodulate_reader(bit_signal, N_samples_per_symbol)
            hex_decoded = get_demodulated_sequence_reader(demodulated_sequence, message_count, hide_demodulated_data)
              
        return hex_decoded