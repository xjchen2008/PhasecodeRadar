import numpy as np
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt
import functions as fn


def wavetable(N, phi = 0):
    #############
    # Parameters
    #############
    c = 3e8
    j = 1j
    T = N / fs  # T=N/fs#Chirp Duration
    # print (N)
    t = np.linspace(0, T, N)
    bw = 10e6  # 20e6#20e6#45.0e5
    fc = 0  # 50e6# 50e6#0e6
    f0 = fc - bw / 2  # -10e6#40e6 # Start Freq
    f1 = fc + bw / 2  # 10e6#60e6# fs/2=1/2*N/T#End freq
    # print('f0 = ',f0/1e6, 'MHz;', 'f1=', f1/1e6, 'MHz')
    k = bw / T  # (f1 - f0) / T
    phi0 = -np.pi / 2  # Phase

    distance = c * freq / k / 2.0  # = c/(2BW), because need an array of distance, so use freq to represent distance.
    win=np.hamming(N)
    # win=np.blackman(N)
    #win = 1
    ##################
    # Create the chirp
    ##################
    y = np.sin(2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation
    yq = np.sin(phi0 + 2 * np.pi * (f0 * t + k / 2 * np.power(t, 2)))  # use this for chirp generation
    y_cx_chirp = y + j * yq
    ##################
    # Create the sine
    ##################
    y_s = np.sin(1*2*np.pi*fs/N*t+ phi)#+ np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_s = np.sin(1*2*np.pi*fs/N*t-np.pi/2 + phi)# + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine1 = y_s + j * yq_s
    fo = 50e6
    y_s2 = np.sin(1*2*np.pi*fo*t)#+ np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_s2 = np.sin(1*2*np.pi*fo*t-np.pi/2)# + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine2 = y_s2 + j * yq_s2

    return np.multiply(y_cx_sine1, win)


def phasecode(M):
    if np.mod(M,2) == 0: # M is even
        L0 = M
    else:
        L0 = 2 * M
    n = np.linspace(0, L0-1, L0)
    phi_1 = np.pi / M  * (k) #+ np.exp(1)/10000
    phi_n =  phi_1 * np.power(n, 2)
    return phi_n


if __name__ == '__main__':
    M = 100  #
    k = 25
    N = int(100 / k) # Base-code length; This determines the range resolution;
    fs = 56e6  # 250e6 #56e6 #1000e6 #250e6  # Sampling freq
    freq = np.fft.fftfreq(N*M, d=1. / fs)
    x = []
    for m in range(0,M):
        phi = phasecode(M)
        x = np.concatenate((x,wavetable(N=N, phi=phi[m])))
        #x = np.concatenate((x, wavetable(N=N, phi=0)))
    #x = wavetable(N = M*N, phi = 0)

    win_all = np.ones(M*N)
    #win_all = np.blackman(M*N)
    x_win = np.multiply(x, win_all)
    pc = fn.PulseCompr(x_win, np.roll(x_win, -10), win = 1, unit='linear')

    #######
    # Plot
    #######
    fn.plot_freq_db(freq, x_win, normalize=False, domain='time')
    plt.title('FFT of Phase-coded Signal')



    plt.figure()
    plt.plot(x_win.real,'*-')
    '''
    plt.title('Phase-coded Signal')
    plt.figure()
    plt.plot(phi, '*-')
    plt.title('Initial Phase')
    '''
    plt.figure()
    fn.plot_freq_db(freq, pc, normalize= True, domain='freq')
    plt.title('Pulse Compression')
    #plt.ylim([-100, 100])
    plt.show()