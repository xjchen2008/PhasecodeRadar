# Xingjian Chen 2021 11 14
# This script generate plots comparing the P4 code and chirp signal pulse compression performance
import numpy as np
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt
import functions as fn
import dsp_filters_BPF
from scipy.signal import hilbert
import main2

if __name__ == '__main__':
    c = 3e8
    N = 1000
    fs = 20e6
    Tp = N/fs
    Rmax1 = 1 / (fs / (N )) * c / 2  # 1/del_F *c/2
    distance = np.linspace(0, Rmax1, N )


    x_chirp = main2.wavetable(Tp=Tp, fs=fs, waveform='chirp')
    x_p4 = main2.wavetable(Tp=Tp, fs=fs, k0=20, waveform='p4')
    win_all = 1
    win_all = np.hanning(N)
    x_win_chirp = np.multiply(x_chirp, win_all)
    x_win_p4 = np.multiply(x_p4, win_all)
    x1 = np.load('waveform_phase_code_chirp_nowin.npy')
    x2 = np.load('waveform_phase_code_chirp_win.npy')
    x3 = np.load('waveform_phase_code_p4_nowin.npy')
    x4 = np.load('waveform_phase_code_p4_win.npy')
    '''
    pc1 = fn.PulseCompr(rx = x1, tx = x1, win = 1, unit='linear')
    pc2 = fn.PulseCompr(rx = x2, tx = x2, win = 1, unit='linear')
    pc3 = fn.PulseCompr(rx = x3, tx = x3, win = 1, unit='linear')
    pc4 = fn.PulseCompr(rx = x4, tx = x4, win = 1, unit='linear')
    '''
    uprate = 10
    delay = 105
    x_chirp_delay = fn.downsampling(np.roll(fn.upsampling(x_chirp, uprate), delay), uprate)
    x_win_chirp_delay = fn.downsampling(np.roll(fn.upsampling(x_win_chirp, uprate), delay), uprate)
    x_p4_delay = fn.downsampling(np.roll(fn.upsampling(x_p4, uprate), delay), uprate)
    x_win_p4_delay = fn.downsampling(np.roll(fn.upsampling(x_win_p4, uprate), delay), uprate)

    pc1 = fn.PulseCompr(rx = x_chirp_delay, tx =x_chirp, win = 1, unit='linear')
    pc2 = fn.PulseCompr(rx = x_win_chirp_delay, tx = x_win_chirp, win = 1, unit='linear')
    pc3 = fn.PulseCompr(rx = x_p4_delay, tx = x_p4, win = 1, unit='linear')
    pc4 = fn.PulseCompr(rx = x_win_p4_delay, tx = x_win_p4, win = 1, unit='linear')
    #fn.plot_freq_db(freq, pc, normalize=True, domain='freq')
    #plt.title('Pulse Compression')

    plt.figure()
    plt.plot((distance), (20 * np.log10(abs(pc1))), 'k*-')  # Matched Filter PC
    plt.plot((distance), (20 * np.log10(abs(pc2))), 'b^-')  # Matched Filter PC
    plt.plot((distance), (20 * np.log10(abs(pc3))), 'rv-')  # Matched Filter PC
    plt.plot((distance), (20 * np.log10(abs(pc4))), 'yo-')  # Matched Filter PC
    # plt.plot(fftshift(distance), fftshift(20 * np.log10(abs(pc))), 'k*-') # PC = mixer method
    plt.xlabel('Distance [m]')
    plt.ylabel('Magnitude [dB]')
    plt.legend(['chirp_nowin','chirp_win','P4_nowin','P4_win'])
    plt.grid()
    plt.xlim([0, 1000])

    plt.figure()
    plt.plot(x_p4.real, '*-')
    plt.title('Phase-coded Signal')
    plt.show()


