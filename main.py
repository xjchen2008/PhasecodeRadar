# Xingjian Chen 2021 Oct 29
# Check out paper "Quadratic Phase Coding for High Duty Cycle Radar Operation"
# https://www.semanticscholar.org/paper/Quadratic-Phase-Coding-for-High-Duty-Cycle-Radar-Mead-Pazmany/3ba09ab4cc5c70833a5de163babf5a5c5e061832
import numpy as np
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt
import functions as fn


def wavetable(N, phi = 0):
    #############
    # Parameters
    #############
    # print (N)
    t = np.linspace(0, Tp-Tp/N, N) # Subpulse duration # T=N/fs
    bw = 10e6  # 20e6#20e6#45.0e5
    fc = 0  # 50e6# 50e6#0e6
    f0 = fc - bw / 2  # -10e6#40e6 # Start Freq
    f1 = fc + bw / 2  # 10e6#60e6# fs/2=1/2*N/T#End freq
    # print('f0 = ',f0/1e6, 'MHz;', 'f1=', f1/1e6, 'MHz')
    k = bw / Tp  # (f1 - f0) / T
    phi0 = -np.pi / 2  # Phase

    distance = c * freq / k / 2.0  # = c/(2BW), because need an array of distance, so use freq to represent distance.
    #win=np.hamming(N)
    #win=np.power(np.blackman(N), 1)
    win = 1
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
    fo = 10e6
    y_s2 = np.sin(1*2*np.pi*fo*t)#+ np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_s2 = np.sin(1*2*np.pi*fo*t-np.pi/2)# + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine2 = y_s2 + j * yq_s2

    return np.multiply(y_cx_sine1, win)


def phasecode(M):
    if np.mod(M,2) == 0: # M is even
        L0 = M
    else:
        L0 = 2 * M
    #n = np.linspace(0, L0-1, L0)
    phi_1 = np.pi / M * (k0)#(k*1000) #+ np.exp(1)/10000
    phi = np.zeros(L0)
    for n in range(0, L0, 1):
        #phi[n] = phi_1 * np.power(n, 2)
        if n<L0/2:
            phi[n] =  phi_1 * np.power(n, 2)
        if n >= L0/2:
            phi[n] = phi_1 * np.power(L0 -n, 2)
    '''
    n = np.linspace(0, L0 - 1, L0)
    phi_n = phi_1 * np.power(n, 2)
    '''
    return phi


if __name__ == '__main__':
    c = 3e8
    j = 1j
    M = 500
    Fp0 = 10e3 # PRF Related to range resolution and range gate. full phase-coded signal is 1ms duration as FMCW SDR radar
    Fp = M * Fp0
    Tp0 = 1 / Fp0
    Tp = 1 / Fp
    k0 = 100 # ralated to freq response
    fs = 60e6
    N = int(Tp * fs)
    # N = 60
    #fs = N / Tp
    d_fs = 1/fs *c /2 # distance of spacing for each sample in time domain
    print('d_fs=',d_fs, '\nfs=', fs/1e6,'MHz', '\nN=',N,
          '\nRange_resolution=',N * d_fs)
    '''
    c = 3e8
    j = 1j
    M = 100  #
    k = 1
    N = int(100/ k) # Base-code length; This determines the range resolution;
    fs = 56e6  # 250e6 #56e6 #1000e6 #250e6  # Sampling freq
    Tp = N / fs  #Sub-pulse Duration
    Tp0 = M*Tp # Tp0 is the full pulse duration; Same as Fp = M* Fp0 in paper(2)
    
    '''
    freq = np.fft.fftfreq(N*M, d=1. / fs)
    del_d = 0.5* c * Tp # Echo trip spacing = range resolution; Sub-pulse duration determines range resolution
    R0 = 0
    del_Rg = del_d * 2
    #Rs = R0 + i * R_del
    #R_mi = Rs + m * del_d
    Rmax = c * Tp0 /2 # Eqn(1) in the paper
    distance = c/2* freq/(M*np.power(Fp0, 2))/k0#np.linspace(-0.5*Rmax, 0.5*Rmax, M*N) #M = Ng?

    x = []
    for m in range(0,M):
        phi = phasecode(M)
        x = np.concatenate((x,wavetable(N=N, phi=phi[m])))
        #x = np.concatenate((x, wavetable(N=N, phi=0)))
    #x = wavetable(N = M*N, phi = 0)

    win_all = 1
    #win_all = np.blackman(M*N)
    x_win = np.multiply(x, win_all)
    pc = fn.PulseCompr(rx = np.roll(x_win, 10), tx = x_win, win = 1, unit='linear')

    #######
    # Plot
    #######

    fn.plot_freq_db(freq, x_win, normalize=False, domain='time')
    plt.title('FFT of Phase-coded Signal')
    plt.figure()
    plt.plot(x_win.real,'*-')

    plt.title('Phase-coded Signal')
    plt.figure()
    plt.plot(phi, '*-')
    plt.title('Initial Phase')

    plt.figure()
    fn.plot_freq_db(freq, pc, normalize= True, domain='freq')
    plt.title('Pulse Compression')

    #plt.xlim([-10, 10])
    #plt.ylim([-100, 100])
    
    plt.figure()
    plt.plot(fftshift(distance), fftshift(20*np.log10(abs(pc))),'k*-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.xlim([-1000, 1000])
    plt.show()