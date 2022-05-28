import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import scipy
from colorama import Style, Fore
from device import Device
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# constants
SUBPORT_FREQ = 847.5e3
NFC_FREQ = 13.56e6
Fc = NFC_FREQ
Fs = SUBPORT_FREQ*2
NFFT_gqrx = NFFT = int(32768)
Fs_gqrx = 1.8e6
RECEIVE_BUFF_SIZE = 16384*8

def load_mag(filename):
    return np.fromfile(filename, np.float32)

def load_gqrx(filename):
    return np.fromfile(filename, np.float32).reshape((-1, 2))

def get_moving_average(spectrum, window_size=512):
    spectrum_mov_avg = []
    for x in spectrum:
        spectrum_mov_avg.append(moving_average(x, window_size, 'valid'))
    return np.array(spectrum_mov_avg)

'''
calculate moving average of the signal
'''
def moving_average(signal, window_size=512, convolution='same'):
    return np.convolve(signal, np.ones(window_size), convolution) / window_size


def get_animated_psd(data,Fs=1.8e6,NFFT=None,Fc=0):
    def animate_psd(i, x, live_interval, Fs, NFFT):
        ## see ax.psd for reference
        pxx, freqs = mlab.psd(x=x[i:int((i+1)*live_interval)], NFFT=NFFT, Fs=Fs, detrend=None,
                                window=None, noverlap=None, pad_to=None,
                                sides='twosided', scale_by_freq=None)
        line[0].set_ydata(10 * np.log10(pxx))
        return line

    live_interval = int(0.01*Fs)
    # based on https://matplotlib.org/stable/gallery/animation/simple_anim.html
    # first plot
    fig, ax = plt.subplots()
    Pxx, freqs, line = ax.psd(data, Fs=Fs, Fc=Fc, sides='twosided', return_line=True, NFFT=NFFT)

    # animation
    ani = animation.FuncAnimation(
        fig, animate_psd, fargs=(data, live_interval, Fs, NFFT), interval=300, blit=True, save_count=int(data.size/live_interval))
    return ani

'''
Plot specgram, cmap='gnuplot2' or cmap='viridis'
'''
def specgram(data, Fs=1.8e6, Fc=13.56e6, NFFT=256, cmap=LinearSegmentedColormap.from_list('paper',["#3F0D12", "red", "#D5BF86", "#F1F0CC", "w"]), ETU=1/(13.56e6/128), N_samples_per_symbol=16, fig_name=None):
    # increase font size
    plt.rcParams['xtick.major.pad']='18'
    plt.rcParams.update({'font.size': 24})

    plt.figure(figsize=(8.2, 4.8))

    spectrum, freqs, t = mlab.specgram(data, Fs=Fs, NFFT=NFFT, sides='twosided')
    # see Welch's average periodogram
    # https://ccrma.stanford.edu/~jos/sasp/Welch_s_Method.html#:~:text=Welch's%20method%20%5B297%5D%20(also,for%20each%20block%2C%20and%20averaging.&text=is%20the%20rectangular%20window%2C%20the,overlapping%20successive%20blocks%20of%20data.
    spectrum = 10*np.log10(np.flip(spectrum.transpose(),axis=0)) # spectrum correction
    
    plt.imshow(spectrum, aspect='auto', interpolation=None, cmap=cmap)

    # help for conversion
    def sample_to_freq(s):
        return s*Fs/NFFT

    def freq_to_sample(f):
        return f/Fs*NFFT

    # adjust x axis
    start_f = int(Fc-Fs/2)
    end_f = int(Fc+Fs/2)
    step_f = Fs/6 # ~ 0.3MHz

    x_step = freq_to_sample(step_f)
    xticks = np.arange(0,NFFT+1,x_step)
    # add central freq sample in the middle
    xticks = np.insert(xticks, int(xticks.size/2), int(NFFT/2))

    xlabels = np.arange(start_f,end_f,step_f)
    # add central freq in the middle
    xlabels = np.insert(xlabels, int(xlabels.size/2), Fc)
    # add last sample
    xlabels = np.append(xlabels, 2*Fc-xlabels[0])
    # round and scale
    xlabels = (xlabels/10**6).round(2)

    plt.xticks(xticks, xlabels, fontsize=18)

    # help for conversion
    def time_to_sample(t):
        return t*Fs/NFFT
    def sample_to_time(s):
        return s/Fs*NFFT

    # adjust y axis
    step_time = 2 #s
    step_sample = time_to_sample(step_time)
    ylabels = np.arange(0,sample_to_time(spectrum.shape[0]),step_time)
    yticks = np.arange(spectrum.shape[0],0,-step_sample)
    plt.yticks(yticks, ylabels)

    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Time [s]')

    plt.tight_layout()
    
    clb = plt.colorbar()
    clb.set_label('Power [dBFS]', labelpad=8)
    if fig_name != None:
        plt.savefig('imgs/profiling/%s' % fig_name, bbox_inches='tight')
    plt.show()
    return spectrum

def specgram_mean(data, message_start=None, message_type=None, NFFT=None):
    assert(NFFT != None)
    data_mean = np.mean(data, axis=1)
    plt.plot(data_mean[::-1])
    # if message_start != None and message_type != None:
    #     message_start = np.array(message_start)
    #     message_type = np.array(message_type)
    #     for x in message_start:
    #         for y in x[message_type==Device.READER]:
    #             v = int(y/(NFFT/2))
    #             plt.plot(v,data_mean[v], 'x')
    plt.show()
    return data_mean

# the baseline specgram mean should be the profiling specgram mean of the blocking card
def find_time(baseline_specgram_mean, action_position):
    return (np.abs(baseline_specgram_mean[::-1] - action_position)).argmin()

def noise_peak_frequencies(spectrum_mov_avg, th_var=4):
    spectrum_var = np.var(spectrum_mov_avg, axis=1)
    noise_freq = np.argmax(spectrum_mov_avg, axis=1)
    noise_freq[noise_freq > int(spectrum_mov_avg.shape[1]/2)] = (int(spectrum_mov_avg.shape[1]/2))*2-noise_freq[noise_freq > int(spectrum_mov_avg.shape[1]/2)]
    noise_freq = np.where(spectrum_var < th_var, 0, noise_freq)
    return noise_freq

def plot_pdf(data, points=100, fig_name=None):
    def multi_gaussian(x, *args):
        m1, m2, s1, s2, k1, k2 = args
        ret = k1*scipy.stats.norm.pdf(x, loc=m1 ,scale=s1)
        ret += k2*scipy.stats.norm.pdf(x, loc=m2 ,scale=s2)
        return ret

    plt.rcParams.update({'font.size': 24})
    
    plt.figure(figsize=(7.6, 4.8))
    n_noise,n_bins_noise,patches = plt.hist(data, bins=points, label="Histogram of samples", density=True)
    
    guess = [0.2, 0.6, 0.1, 0.1, 1, 1]
    popt, pcov = scipy.optimize.curve_fit(multi_gaussian, n_bins_noise[:n_noise.size], n_noise, guess, maxfev=5000)
    plt.plot(n_bins_noise[:n_noise.size], multi_gaussian(n_bins_noise[:n_noise.size], *popt), 'r--', linewidth=2, label='Fit')

    # fitting
    # mu, sigma = scipy.stats.norm.fit(data)
    # best_fit_line = scipy.stats.norm.pdf(n_bins_noise, mu, sigma)
    # plt.plot(n_bins_noise, best_fit_line)

    # plt.ylim(0,np.max(n_noise[20:]+1))
    # plt.xlim(0.05,)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Probability Density")

    if fig_name != None:
        plt.savefig('imgs/profiling/%s' % fig_name, bbox_inches='tight')
    plt.show()

'''
update crc calculation
'''
def update_crc(ch, lpw_crc):
    ch = ch ^ lpw_crc & 0x00ff
    ch = ch ^ (ch << 4) & 0xff # added & 0xff for python version
    return (lpw_crc >> 8) ^ (ch << 8) ^ (ch << 3) ^ (ch >> 4)

'''
See ISO 1443-3 appendix for implementation
@data is a bytearray
'''
def crc_a(data):
    w_crc = 0x6363 # ITU-V.41 for CRC_A
    for b in data:
        w_crc = update_crc(b, w_crc)
        
    transmit_first = w_crc & 0xff
    transmit_second = (w_crc >> 8) & 0xff    
    return bytearray([transmit_first, transmit_second])

'''
Find pattern which matches better
'''
def pattern_match(pattern_to_match, *sequences):
    highest_correlation = np.sum(pattern_to_match == np.array(sequences[0]))
    matched_pattern = sequences[0]
    for sequence in sequences[1:]:
        correlation = np.sum(pattern_to_match == np.array(sequence))
        if correlation > highest_correlation:
            highest_correlation = correlation
            matched_pattern = sequence
    return matched_pattern

# '''
# Calculates the moving average given a window of size window_size
# '''
# def moving_average_rt(first_window_element, new_sample, moving_sum_value, count, window_size):
    
#     if count >= window_size:
#         moving_sum_value -= first_window_element # remove older elements
#         count -= 1
#     ## fill with new sample
#     moving_sum_value += new_sample
#     count += 1
        
#     current_moving_average = moving_sum_value / count
    
#     return current_moving_average, moving_sum_value, count

# '''
# Calculates window minimum based on https://afteracademy.com/blog/sliding-window-maximum
# '''
# # https://afteracademy.com/blog/sliding-window-maximum
# def window_min(i, q, data, window_size):
#     if i >= window_size - 1:# remove elements out of window
#         while(len(q) != 0 and q[0] <= i - window_size):
#             q.popleft()
            
#     while(len(q) != 0 and data[i] < data[q[len(q) - 1]]):
#         q.pop()
#     q.append(i)
#     return data[q[0]], q

# '''
# Calculates window maximum based on https://afteracademy.com/blog/sliding-window-maximum
# '''
# # https://afteracademy.com/blog/sliding-window-maximum
# def window_max(i, q, data, window_size):
#     if i >= window_size - 1:# remove elements out of window (those on left of q)
#         while(len(q) != 0 and q[0] <= i - window_size):
#             q.popleft()
            
#     # is the element added greater than old greatest?
#     while(len(q) != 0 and data[i] >= data[q[len(q) - 1]]):
#         q.pop()
#     q.append(i)
#     return data[q[0]], q

# '''
# Calculates window gradient taking the difference between minimum and maximum
# '''
# def window_gradient(i, q_min, q_max, data, window_size):
#     min_gradient, q_min = window_min(i, q_min, data, window_size)
#     max_gradient, q_max = window_max(i, q_max, data, window_size)
#     gradient = max_gradient - min_gradient
#     return gradient, q_min, q_max

def get_libnfc_sequence(file_name):
    message_expected = []
    if file_name != None:
        for x in pd.read_csv(file_name, header=None, skip_blank_lines=False).values:
            message_expected.append(x[0].rstrip() if isinstance(x[0], str) else None)
        # message_expected.pop()
    return message_expected

def get_mifare_classic_sequence(repeat_cmd=1, repeat_blck=80):
    wupa = "52"
    select_all = "93 20"
    select_tag = "93 70 33 2c 51 24 6a d6 59"
    bad = "29 88 cb e2"
    halt = "50 00 57 cd"

    wupa_response = "04 00"
    select_all_response = "33 2c 51 24 6a"
    select_tag_response = "08 b6 dd"

    # commands running on apdu_get_data
    cmd_sequence = [] #known
    resp_sequence = [] #unknown by attacker
    sequence = [] #unknown by attacker
    for k in range(repeat_blck):
        cmd_sequence.append(wupa)
        sequence.append(cmd_sequence[-1])

        resp_sequence.append(wupa_response)
        sequence.append(resp_sequence[-1])

        cmd_sequence.append(select_all)
        sequence.append(cmd_sequence[-1])

        resp_sequence.append(select_all_response)
        sequence.append(resp_sequence[-1])

        cmd_sequence.append(select_tag)
        sequence.append(cmd_sequence[-1])

        resp_sequence.append(select_tag_response)
        sequence.append(resp_sequence[-1])
        if k != repeat_blck - 1:
            cmd_sequence.append(bad)
            sequence.append(cmd_sequence[-1])

            resp_sequence.append(None)
            sequence.append(resp_sequence[-1])
    cmd_sequence.append(halt)
    sequence.append(cmd_sequence[-1])
    resp_sequence.append(None)
    sequence.append(resp_sequence[-1])

    return cmd_sequence, resp_sequence, sequence