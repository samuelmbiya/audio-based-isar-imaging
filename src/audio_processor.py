# --------------------------------------------------
# Audio signal processing and result plotting source code
# --------------------------------------------------
# Author: Samuel Mbiya
# Last modified: 23 June 2024

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from scipy import signal
from scipy.fft import fftshift
from scipy.signal import find_peaks

import argparse
# from pathlib import Path

parser = argparse.ArgumentParser()

# $range $center_freq $bandwidth $num_pulses
parser.add_argument("range", type=float)
parser.add_argument("center_freq", type=float)
parser.add_argument("bandwidth", type=float)

args = parser.parse_args()

c = 343
r_max = args.range#3

fc = args.center_freq
B = args.bandwidth
f_start = fc -B/2
f_end = fc + B/2

R_res = c/(2*B)

fs = 44100
ts = 1/fs

PRI = 2*r_max/c
print("PRI: ", PRI)
pulse_width = 0.5*PRI

t_axis = np.linspace(0,PRI, int(PRI/ts))

f_axis = np.linspace(-fs/2,fs/2,len(t_axis))

def load_recording(path):
    '''
    Load and return the received signal .wav recording
    
    '''
    received_sig, fs = librosa.load(path, sr=44100)  
    return received_sig

def real_to_complex(received_signal):
    '''
    Convert the real valued recorded signal to complex signal
    '''
    return signal.hilbert(received_signal)

def matched_filter(transmitted_chirp, received_signal):
    '''
    Perform matched filter process using the received signal and transmitted chirp signal
    '''
    matched_filter_out = signal.correlate(received_signal, signal.hilbert(transmitted_chirp), 'same')
    lags = signal.correlate(len(received_signal.real), len(transmitted_chirp))
    matched_filter_out /= np.max(matched_filter_out)
    return matched_filter_out

def generate_range_profiles_matrix(peaks, matched_filter_out, PRI):
    '''
    Generate and return the range profile matrix
    '''    
    range_profile_idxs= []
    for x in range(0,len(peaks)-1):
        if peaks[x+1]-peaks[x] >= (fs*PRI)*(1-0.3):
            range_profile_idxs.append([peaks[x], peaks[x+1]])

    num_pulses = len(range_profile_idxs)
    num_range_bins = np.min(np.diff(range_profile_idxs))

    range_profiles_matrix = np.zeros(shape=(num_pulses, num_range_bins),dtype=np.complex_)

    for pulse_num in range(0, num_pulses):
        range_profiles_matrix[pulse_num] = matched_filter_out[range_profile_idxs[pulse_num][0]:range_profile_idxs[pulse_num][0]+num_range_bins]
    
    return range_profiles_matrix, num_pulses, num_range_bins

def generate_range_doppler_map(range_profiles_matrix, num_pulses, num_range_bins):
    '''
    Generate and return the range doppler map, given the range profiles matrix as input
    '''    
    range_dopp_map = np.zeros(shape=(num_pulses, num_range_bins))
    for x in range(0,num_range_bins):
        range_dopp_map[:,x] = np.fft.fftshift(20*np.log10(np.abs(np.fft.fft(range_profiles_matrix[:,x]*signal.windows.hann(num_pulses)))))
    return range_dopp_map
    
def ca_cfar(range_dopp_map, guard_cells, window_size, threshold):
    '''
    Perform CFAR detection on range doppler map
    '''
    num_pulses = np.shape(range_dopp_map)[0]
    num_range_bins = np.shape(range_dopp_map)[1]

    cfar_map = np.zeros(shape=(num_pulses, num_range_bins))

    P_fa = 0.0001

    for i in range(0,num_pulses):
        for j in range(0,num_range_bins):
            cfar_window = []
            cut = range_dopp_map[i][j]
            lindx = np.arange(int(np.maximum(0, j - (window_size)/2)), int(j - guard_cells/2))

            rindx = np.arange(int(j + guard_cells/2), int(np.minimum(num_range_bins, j + (window_size)/2)))

            cfar_window = np.concatenate((range_dopp_map[i][lindx], range_dopp_map[i][rindx])) 
            alpha = np.power(P_fa, -1/len(cfar_window)) - 1
            # threshold = alpha*np.sum(cfar_window)/len(cfar_window)

            cfar_map[i][j] = range_dopp_map[i][j] > threshold#threshold

    return cfar_map

def align_range_profiles(range_profiles_matrix):
    N, M = range_profiles_matrix.shape
    aligned_range_profiles = np.zeros_like(range_profiles_matrix)
    reference_range_profile = range_profiles_matrix[0, :]

    for i in range(1, N):
        current_profile = range_profiles_matrix[i,:]

        # range alignment
        correlation = signal.correlate(reference_range_profile, current_profile, mode='full')
        lag = np.argmax(correlation) - (M - 1)

        # phase adjustment
        phase_diff = np.angle(reference_range_profile) - np.angle(current_profile)
        mean_phase_diff = np.mean(phase_diff)

        if lag > 0:
            aligned_range_profiles[i, lag:] = current_profile[:M-lag]* np.exp(1j * mean_phase_diff)
        else:
            aligned_range_profiles[i, :M+lag] = current_profile[-lag:]* np.exp(1j * mean_phase_diff)
    return aligned_range_profiles

def mti_filter(range_profiles_matrix, nPulseCancel):
    # nPulseCancel = 3
    mti_filter = np.transpose(-np.ones(nPulseCancel)/nPulseCancel)
    midIdx = int(((nPulseCancel+1)/2))
    mti_filter[midIdx-1] = mti_filter[midIdx] + 1
    mti_filter = np.reshape(mti_filter, (nPulseCancel,1))

    range_profiles_matrix = signal.fftconvolve(range_profiles_matrix, mti_filter, mode='same')
    return range_profiles_matrix

# plot output of matched filter
def plot_matched_filter_output():
    N_samples = len(received_signal)  
    T = N_samples/fs 

    t_received = np.linspace(0,T, N_samples)
    t_start = 0
    t_end = T

    fig1, (ax_orig, ax_corr) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    ax_orig.plot(t_received[round(t_start*fs):round(t_end*fs)], received_signal.real[round(t_start*fs):round(t_end*fs)])
    ax_orig.set_title('Original signal')
    ax_orig.grid()

    ax_corr.plot(t_received[round(t_start*fs):round(t_end*fs)], 20*np.log10(np.abs(matched_filter_output[round(t_start*fs):round(t_end*fs)])))
    ax_corr.set_title('Cross-correlated signal')
    ax_corr.grid()

# plots to confim peak positions in signal returns 
def plot_peak_finding_output():
    plt.figure()
    plt.plot(20*np.log10(np.abs(matched_filter_output)))
    plt.plot(peaks, 20*np.log10(np.abs(matched_filter_output))[peaks], "x")
    plt.plot(np.zeros_like(20*np.log10(np.abs(matched_filter_output))), "--", color="gray") 

# plot range profiles matrix
def plot_range_profiles_matrix():
    plt.figure()
    plt.imshow(20*np.log10(np.abs(range_profiles_matrix)),interpolation='nearest', aspect='auto', cmap='jet')
    plt.title('Range Profile Matrix')
    plt.xlabel('Fast time - Range, m')
    plt.ylabel('Slow time - Pulse #')

def plot_small_integration_angle_range_doppler_map():
    # plot range doppler map
    fig2, ax2 = plt.subplots(1,1)
    img = plt.imshow(np.clip(range_doppler_map, -60,10), interpolation='nearest', aspect='auto', cmap='jet')
    ax2.set_xticks(np.linspace(0,(Nr), 12))
    ax2.set_xticklabels(["%.2f" % x for x in np.linspace(0,(Nr)/fs * c/2, 12)])
    ax2.set_yticks(np.linspace(0,Np-1,Np))
    # ax2.set_yticks(np.linspace(Np/2,Np/2 -1,Np))
    ax2.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * wavelength/(2*angular_velocity) ,1/(2*PRI) * wavelength/(2*angular_velocity), Np)])
    # ax2.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * c/(2*fc) ,1/(2*PRI) * c/(2*fc), Np)])
    # ax2.set_yticklabels(["%.2f" % x for x in np.linspace(-(Np/2) *R_c ,(Np/2) * R_c, Np)])
    # plt.title('Range Doppler Map')
    plt.xlabel('Slant range, m')
    plt.ylabel('Cross range, m')
    # plt.ylim(25,35)
    plt.colorbar().set_label('Signal level, dB')

    # plot range doppler map
    fig7, ax7 = plt.subplots(1,1)
    img = plt.imshow(np.clip(range_doppler_map, -60,10), interpolation='nearest', aspect='auto', cmap='jet')
    ax7.set_xticks(np.linspace(0,(Nr), 12))
    ax7.set_xticklabels(["%.2f" % x for x in np.linspace(0,(Nr)/fs * c/2, 12)])
    ax7.set_yticks(np.linspace(0,Np-1,Np))
    # ax2.set_yticks(np.linspace(Np/2,Np/2 -1,Np))
    # ax7.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * wavelength/(2*angular_velocity) ,1/(2*PRI) * wavelength/(2*angular_velocity), Np)])
    ax7.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * c/(2*fc) ,1/(2*PRI) * c/(2*fc), Np)])
    # ax2.set_yticklabels(["%.2f" % x for x in np.linspace(-(Np/2) *R_c ,(Np/2) * R_c, Np)])
    # plt.title('Range Doppler Map')
    plt.xlabel('Slant range, m')
    plt.ylabel('Velocity, m/s')
    # plt.ylim(25,35)
    plt.colorbar().set_label('Signal level, dB')    

def plot_wide_integration_angle_range_doppler_map():
    # plot range doppler map
    fig2, ax2 = plt.subplots(1,1)
    img = plt.imshow(np.clip(range_doppler_map, -30, 0), interpolation='nearest', aspect='auto', cmap='jet')
    ax2.set_xticks(np.linspace(0,(Nr), 12))
    ax2.set_xticklabels(["%.2f" % x for x in np.linspace(0,(Nr)/fs * c/2, 12)])
    ax2.set_yticks(np.linspace(0,Np-1,Np//20))
    ax2.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * wavelength/(2*angular_velocity) ,1/(2*PRI) * wavelength/(2*angular_velocity), Np//20)])
    # plt.title('Range Doppler Map')
    plt.xlabel('Down Range, m')
    plt.ylabel('Cross range, m')
    plt.colorbar().set_label('Signal level, dB')

    fig6, ax6 = plt.subplots(1,1)
    img = plt.imshow(np.clip(range_doppler_map, -30, 0), interpolation='nearest', aspect='auto', cmap='jet')
    ax6.set_xticks(np.linspace(0,(Nr), 12))
    ax6.set_xticklabels(["%.2f" % x for x in np.linspace(0,(Nr)/fs * c/2, 12)])
    ax6.set_yticks(np.linspace(0,Np-1,Np//20))
    ax6.set_yticklabels(["%.2f" % x for x in np.linspace(-1/(2*PRI) * c/(2*fc) ,1/(2*PRI) * c/(2*fc), Np//20)])
    # plt.title('Range Doppler Map')
    plt.xlabel('Range, m')
    plt.ylabel('Velocity, m/s')
    plt.colorbar().set_label('Signal level, dB')

def plot_mean_signal_level_of_range_profiles():
    fig4, ax4 =  plt.subplots(1,1)
    range_profiles = [x for x in np.transpose(range_doppler_map)]
    p = plt.plot(np.mean(range_profiles, axis=1))
    ax4.set_xticks(np.linspace(0,(Nr), 12))
    ax4.set_xticklabels(["%.2f" % x for x in np.linspace(0,(Nr)/fs * c/2, 12)])
    plt.title('Signal Level vs Range')
    plt.xlabel('Range, m')
    plt.ylabel('Signal level, dB')

if __name__ == "__main__":

    # Load signal

    received_signal = load_recording('data/output.wav')

    # Conversion of Real Signal to Complex Signal

    received_signal_iq = real_to_complex(received_signal)

    n = np.arange(fs * pulse_width)
    transmitted_chirp = (np.sin(2 * np.pi  * (B/pulse_width * n*1/fs + f_start) / fs * n)).astype(np.float32)    

    N_samples = len(transmitted_chirp)
    T = N_samples/fs

    t = np.linspace(0,T, N_samples)

    # Pulse Compression
    matched_filter_output = matched_filter(transmitted_chirp, received_signal_iq)

    # Peak finding
    matched_filter_peak_threshold_dB = -1.6#-0.1#-0.0069#-0.00689
    peaks, _ = find_peaks(20*np.log10(np.abs(matched_filter_output)), height=matched_filter_peak_threshold_dB)

    # Range Profile Matrix
    range_profiles_matrix, Np, Nr = generate_range_profiles_matrix(peaks, matched_filter_output, PRI)

    # Range Alignment and Phase adjustment
    range_profiles_matrix = align_range_profiles(range_profiles_matrix)

    # MTI filter
    nPulseCancel = 3
    range_profiles_matrix = mti_filter(range_profiles_matrix, 3)

    wavelength = c/args.center_freq #Hz
    angular_velocity = 0.1868327478#(1.784121 RPM)0.2067517376#0.1693580945 #rad/s
    CPI = Np*PRI
    print("CPI: ", CPI, "s")
    R_c = 0.5 * wavelength/(angular_velocity*CPI)
    print("Cross range resolution: ", R_c, "m")
  
    wide_integration_angle_processing = True

    # Range Doppler Processing
    if wide_integration_angle_processing == False:
        print(Np,Nr)
        Np_total = 900
        Np = 43
        CPI = Np*PRI
        print("CPI: ", CPI, "s")
        R_c = 0.5 * wavelength/(angular_velocity*CPI)
        print("Cross range resolution: ", R_c, "m")
        start = 700
        for index in np.arange(60):
            print("hello")
            # Np = 53
            range_doppler_map = generate_range_doppler_map(range_profiles_matrix[(start+index):(start+Np+index),:], Np, Nr)

            plot_small_integration_angle_range_doppler_map()
    else:
        range_doppler_map = generate_range_doppler_map(range_profiles_matrix[(0):(Np),:], Np, Nr)

        plot_wide_integration_angle_range_doppler_map()

# plot_matched_filter_output()
# plot_peak_finding_output()
# plot_range_profiles_matrix()
# plot_mean_signal_level_of_range_profiles()

plt.show()

