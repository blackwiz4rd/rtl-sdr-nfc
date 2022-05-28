from time import time
from nfc_signal_offline import *

def get_demodulation_stats(gnuradio_file, libnfc_file, expected_file, output_file, stats_file, mean_samples=10):
    # load gqrx signal
    data = load_mag(gnuradio_file) + 0.3
    start = time()
    s = NfcSignal(data, 
                  expected_file=expected_file, 
                  libnfc_file=libnfc_file, 
                  output_file=output_file,
                  attack_mode=0, 
                  mean_samples=mean_samples, 
                  message_batch_size=8)
    end = time()
    # print(f"init duration {end-start}").pop()
    s.start_demodulation()
    # s.save_hex()
    demodulation_stats = s.demodulation_stats() # cool
    display(demodulation_stats)
    demodulation_stats.to_csv(stats_file, index=False)