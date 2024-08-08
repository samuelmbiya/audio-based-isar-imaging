# --------------------------------------------------
# Signal generation and transmission source code
# --------------------------------------------------
# Author: Samuel Mbiya
# Last modified: 23 June 2024

import time

import numpy as np
import pyaudio

from scipy import signal
from scipy.fft import fft, fftshift

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("range", type=float)
parser.add_argument("center_freq", type=float)
parser.add_argument("bandwidth", type=float)
parser.add_argument("num_pulses", type=int)

args = parser.parse_args()

p = pyaudio.PyAudio()

c = 343
volume = 1 
fs = 44100 
PRI = (2*args.range)/c 
print("PRI: ", PRI)
pulse_width = PRI*0.5 # in seconds, may be float

fc = args.center_freq

B = args.bandwidth

f_start = fc -B/2
f_end = fc + B/2

sample_no = np.arange(fs * 1*pulse_width)

num_pulses_per_burst = args.num_pulses

# LFM chirp pulse signal definition
transmitted_signal = np.sin(2 * np.pi  * (B/pulse_width * sample_no*1/fs + f_start) / fs * sample_no).astype(np.float32)

# generate pulse burst
samples = []
for x in range(1,num_pulses_per_burst+1):
    samples = np.hstack((samples, transmitted_signal))
    samples = np.hstack((samples, 0*transmitted_signal)).astype(np.float32)

output_bytes = (volume * samples).tobytes()

# print(output_bytes)

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

time.sleep(0.5)

start_time = time.time()

# transmit pulse burst
stream.write(output_bytes)
print("Played sound for {:.2f} seconds".format(time.time() - start_time))

stream.stop_stream()
stream.close()

p.terminate()