# Xingjian Chen 2021 Oct 29
# Check out paper "Quadratic Phase Coding for High Duty Cycle Radar Operation"
# https://www.semanticscholar.org/paper/Quadratic-Phase-Coding-for-High-Duty-Cycle-Radar-Mead-Pazmany/3ba09ab4cc5c70833a5de163babf5a5c5e061832
import numpy as np
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt
import functions as fn
import dsp_filters_BPF
from scipy.signal import hilbert
import coe_wavetable_4096 as coe
from readbin import readbin2

plt.rc('font', size=15)


def wavetable_ref(N, win=True, phi=0):
    n = np.linspace(0, N - 1, N)
    y_p3 = np.sin(np.pi / N * np.power(n, 2))  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_p3 = np.sin(np.pi / N * np.power(n, 2) - np.pi / 2)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine3 = y_p3 + j * yq_p3
    y_p4 = np.sin(k0 * np.pi / N * n * (
                n - N))  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. k0 is related to the cycle/ period of the phase coding signal
    yq_p4 = np.sin(k0 * np.pi / N * n * (n - N) - np.pi / 2)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine4 = y_p4 + j * yq_p4
    uprate_ = 1
    y_cx_sine4_uprate = fn.upsampling(y_cx_sine4, uprate_)
    y_cx_sine4_delay = np.roll(y_cx_sine4_uprate, 0) \
                       + np.roll(y_cx_sine4_uprate, 1)  # \
    # + np.roll(y_cx_sine4_uprate, 2) \
    # + np.roll(y_cx_sine4_uprate, 4) + np.roll(y_cx_sine4_uprate, 8) \
    # + np.roll(y_cx_sine4_uprate, 32) + np.roll(y_cx_sine4_uprate, 64) \
    # + np.roll(y_cx_sine4_uprate, 128)
    y_cx_woo = fn.downsampling(y_cx_sine4_delay, uprate_)

    if win == True:
        # win = np.power(np.hanning(N), 1)
        win = np.abs(np.sin(np.pi * (-n + 0.5) / N))
    else:
        win = 1
    return np.multiply(y_cx_woo, win)


def wavetable(N, win=True, phi=0):
    #############
    # Parameters
    #############
    # print (N)
    # t = np.linspace(0, Tp-Tp/N, N) # Subpulse duration # T=N/fs # No the last point
    t = np.linspace(0, Tp, N)  # With the last point
    n1 = np.linspace((3 * N / 4), N - 1, int(N / 4))
    n2 = np.linspace(0, (1 * N / 4 - 1), int(N / 4))
    # n = np.concatenate([n2, (1.0+1.0*j)*np.zeros(int(N/2)), n1])
    # k0=1
    n = np.linspace(0, N - 1, N)
    fc = 0 #15e6  # 50e6# 50e6#0e6
    f0 = fc - bw / 2  # -10e6#40e6 # Start Freq
    f1 = fc + bw / 2  # 10e6#60e6# fs/2=1/2*N/T#End freq
    # print('f0 = ',f0/1e6, 'MHz;', 'f1=', f1/1e6, 'MHz')
    k = (f1 - f0) / Tp #bw / Tp  #
    phi0 = -np.pi / 2  # Phase

    # distance = c * freq / k / 2.0  # = c/(2BW), because need an array of distance, so use freq to represent distance.
    # win=np.hamming(N)
    # win=np.power(np.blackman(N), 2)
    if win == True:
        win_con = []
        win_0 = np.power(np.hanning(N / k0), 1)
        # win_0 = np.power(np.blackman(N / k0), 1)
        for idx in range(k0):
            win_con = np.concatenate((win_con, win_0))
        win = win_con
        #win = 1
        # win=np.power(np.hanning(N), 1)
        # win = np.abs(np.sin(np.pi*(-n+0.5)/N))
    else:
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
    # phase coding
    y_s = np.sin(1 * 2 * np.pi * fs / N * t + phi)  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_s = np.sin(1 * 2 * np.pi * fs / N * t - np.pi / 2 + phi)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    # No phase coding
    # y_s = np.sin(1 * 2 * np.pi * fs / N * t )  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    # yq_s = np.sin(1 * 2 * np.pi * fs / N * t - np.pi / 2 )  # + np.sin(4*np.pi*fs/N*t-np.pi/2)

    y_cx_sine1 = y_s + j * yq_s
    fo = 10e6
    y_s2 = np.sin(1 * 2 * np.pi * fo * t)  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_s2 = np.sin(1 * 2 * np.pi * fo * t - np.pi / 2)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine2 = y_s2 + j * yq_s2
    ##########################
    y_p3 = np.sin(k0 * np.pi / N * np.power(n, 2))  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. The
    yq_p3 = np.sin(k0 * np.pi / N * np.power(n, 2) - np.pi / 2)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine3 = y_p3 + j * yq_p3

    # F = fs/N * N/20
    # omega = 2*np.pi*F
    # y_p4 = np.sin(omega * t + k0*np.pi/N*n*(n-N))  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. k0 is related to the cycle/ period of the phase coding signal
    # yq_p4 = np.sin(omega*t -np.pi/2 + k0*np.pi/N*n*(n-N))  # + np.sin(4*np.pi*fs/N*t-np.pi/2)

    y_p4 = np.sin(k0 * np.pi / N * n * (
                n - N))  # + np.sin(4*np.pi*fs/N*t)# just use LO to generate a LO. k0 is related to the cycle/ period of the phase coding signal
    yq_p4 = np.sin(k0 * np.pi / N * n * (n - N) - np.pi / 2)  # + np.sin(4*np.pi*fs/N*t-np.pi/2)
    y_cx_sine4 = y_p4 + j * yq_p4
    uprate_ = 1
    y_cx_sine4_uprate = fn.upsampling(y_cx_sine4, uprate_)
    y_cx_sine4_delay = np.roll(y_cx_sine4_uprate, 0) \
                       + np.roll(y_cx_sine4_uprate, 1)  # \
    # + np.roll(y_cx_sine4_uprate, 2) \
    # + np.roll(y_cx_sine4_uprate, 4) + np.roll(y_cx_sine4_uprate, 8) \
    # + np.roll(y_cx_sine4_uprate, 32) + np.roll(y_cx_sine4_uprate, 64) \
    # + np.roll(y_cx_sine4_uprate, 128)
    y_cx_woo = fn.downsampling(y_cx_sine4_delay, uprate_)

    # y_cx_pad = np.pad(np.multiply(y_cx_sine1, win),180,'constant') # for N= 40
    # y_cx_pad = np.pad(np.multiply(y_cx_sine1, win),190,'constant') # for N= 20
    #y_cx_pad = np.pad(np.multiply(y_cx_chirp, win), int((4000-N)/2), 'constant')  # for N= 20
    N_zero = int((1000-N))
    y_cx_pad = np.concatenate((np.zeros(600), np.multiply(y_cx_woo, win), np.zeros(200))) # the first 600 points cannot tx in X310, so zeros first.
    noise = np.random.normal(0, 1e-3, len(y_cx_pad))
    # plt.plot(phase_mod,'o-')
    # plt.show()
    return np.multiply(y_cx_woo, win) #y_cx_pad #y_cx_pad  # np.multiply(y_cx_sine1, win) #y_cx_sine1 #np.multiply(y_cx_chirp, win) #np.multiply(y_cx_sine3, y_cx_sine3) #y_cx_sine4 #y_cx_pad +noise


def phasecode(M):
    if np.mod(M, 2) == 0:  # M is even
        L0 = M
    else:
        L0 = 2 * M
    # n = np.linspace(0, L0-1, L0)
    phi_1 = np.pi / M * (k0)  # (k*1000) #+ np.exp(1)/10000
    phi = np.zeros(L0)
    '''
    for n in range(0, L0, 1):
        #phi[n] = phi_1 * np.power(n, 2)
        if n<L0/4:
            phi[n] =  phi_1 * np.power(n, 2)
        if L0/4 <= n < L0/2:
            phi[n] = phi_1 * np.power(L0/2 -n, 2)
        if L0/2 <= n < 3*L0/4:
            phi[n] =  phi_1 * np.power(n-L0/2, 2)
        if n >= L0 *3/ 4:
            phi[n] = phi_1 * np.power(L0 -n, 2)
    '''

    # Previous initial phase
    for n in range(0, L0, 1):
        # phi[n] = phi_1 * np.power(n, 2)
        # paper or
        if n < L0 / 2:
            phi[n] = phi_1 * np.power(n, 2)
        if n >= L0 / 2:
            phi[n] = phi_1 * np.power(L0 - n, 2)
        # chirp order
        # if n<L0/2:
        #    phi[n] =  phi_1 * np.power(L0/2-n, 2)
        # if n >= L0/2:
        #    phi[n] = phi_1 * np.power(n-L0

    '''
    n = np.linspace(0, L0 - 1, L0)
    phi_n = phi_1 * np.power(n, 2)
    '''
    # phi = np.zeros(L0)
    return phi


def test():
    fn.plot_freq_db(freq, x_win, normalize=False, domain='time')
    # plt.plot(fftshift(freq), fftshift(20*np.log10(abs(np.fft.fft(x_win,axis=0)))))
    plt.title('FFT of Phase-coded Signal')
    plt.figure()
    plt.plot(x_win.real, '*-')


if __name__ == '__main__':
    c = 3e8
    j = 1j
    uprate = 2
    roll = 2 * 20 * 3 + 21  # 40*20+1
    a = .1#40 # .1#40
    M = int(10 * a)  # int(50 /a)  # tune with Fp0 to increase range gate or range ambiguity
    Fp0 = 5e3 * 100/5/ a*1   # 16e3 * a # PRF Related to range resolution and range gate. full phase-coded signal is 1ms duration as FMCW SDR radar
    Fp = M * Fp0
    Tp0 = 1 / Fp0
    Tp = 1 / Fp
    k0 = 1  # ralated to freq response
    fs = 20e6
    N = int(Tp * fs)  # 20

    bw = 200e6  # 56e6  # FMCW chirp bandwidth 20e6#20e6#45.0e5
    print('bw=', bw)
    '''N = 60
    fs = N / Tp'''
    d_fs = 1 / fs * c / 2  # The "unit resolution" from unit sample time; distance of spacing for each sample in time domain
    Rmax = c * Tp0 / 2  # Eqn(1) in the paper
    print('d_fs=', d_fs, '\nfs=', fs / 1e6, 'MHz', '\nN=', N, 'M=', M, ',Spectrum Spacing: fs/N=', fs / 1e6 / N, 'MHz,',
          '\nPulse_Radar_Range_resolution=', N * d_fs, ';FMCW_Radar_Range_resolution=', c / (2 * bw),
          '\nRmax=', Rmax)

    del_d = 0.5 * c * Tp  # Echo trip spacing = range resolution; Sub-pulse duration determines range resolution
    R0 = 0
    del_Rg = del_d * 2
    # Rs = R0 + i * R_del
    # R_mi = Rs + m * del_d

    x = []
    for m in range(0, M):
        phi = phasecode(M)
        x = np.concatenate((x, wavetable(N=N, win=False, phi=phi[m])))
        # x = np.concatenate((x, coe.y_cx))
    # x = wavetable(N = M*N, phi = 0)
    # x = readbin2("usrp_samples.dat", np.short, len(x), 0)
    freq = np.fft.fftfreq(len(x), d=1. / fs)

    Rmax1 = len(x) * c / (2 * fs)  # 1/del_F *c/2
    distance = np.linspace(0, Rmax1, len(x))  # FMCW PC=Matched Filter radar
    # distance = c / 2 * freq / (bw / Tp)  # FMCW radar PC=Stretch Method
    #distance = 2 * c * freq / (
    #            M * np.power(Fp0, 2)) / k0  # phase coding radar np.linspace(-0.5*Rmax, 0.5*Rmax, M*N) #M = Ng?
    win_all = 1
    # win_all = np.blackman(M*N)
    x_win = np.multiply(x, win_all)
    x_win_uprate = fn.upsampling(x_win, uprate)
    x_win_uprate_roll = np.roll(x_win_uprate, roll)
    x_win_delay = fn.downsampling(x_win_uprate_roll, uprate)
    x_win_delay = x_win  # + x_win_delay #np.roll(x_win, roll) #

    # test()
    #'''
    #
    # x_win_BPF_real = dsp_filters_BPF.run(x_win.real,fs=fs, highcut=15e6, lowcut=5e6)
    # x_win_BPF = hilbert(x_win_BPF_real)
    # x_win = x_win_BPF
    # x_win_0 = np.concatenate((np.array([0]), x_win[0:-1]))
    # x_win_0 = x_win[0:-1]
    x_win_0 = wavetable_ref(N=N - 1, win=True, phi=phi[m])
    x_win_0_full = np.concatenate((np.array([0]), x_win_0))
    pc = fn.PulseCompr(rx=x_win_delay, tx=x_win, win=1, unit='linear')
    # pc = fn.PulseCompr(rx = hilbert(x_win_delay.real), tx = hilbert(x_win.real), win = 1, unit='linear')
    # pc = fn.PulseCompr(rx=x_win_delay.real, tx=x_win.real, win=1, unit='linear')
    # np.save(file='data/pc.npy', arr=pc)
    #######
    # Plot
    #######

    fn.plot_freq_db(freq, x_win_delay, normalize=True, domain='time')
    # plt.plot(fftshift(freq)/1e6, fftshift(20*np.log10(1e-1+abs(np.fft.fft(x_win,axis=0)))))
    # plt.plot(fftshift(20 * np.log10(1e-1+abs(np.fft.fft(x_win, axis=0))))) # for padded signal
    # plt.plot(fftshift(20 * np.log10(abs(np.fft.fft(x_win, axis=0))))) # for non-padded signal
    plt.title('FFT of Phase-coded Signal')
    plt.figure()
    plt.plot(x_win_delay.real, '*-')
    plt.title('Phase-coded Signal')
    plt.figure()
    plt.plot(phi, '*-')
    plt.title('Initial Phase')
    
    #plt.figure()
    #fn.plot_freq_db(freq, pc, normalize= True, domain='freq')
    #plt.title('Pulse Compression')
    
    # plt.xlim([-10, 10])
    # plt.ylim([-100, 100])
    pc_woo_nowin = np.load('data/pc_woo_noWin_integer.npy')
    pc_woo_nowin_db = (20 * np.log10(abs(pc_woo_nowin)))
    pc_woo_nowin_db_normalized = pc_woo_nowin_db - pc_woo_nowin_db.max()

    pc_woo = np.load('data/pc_woo_Win_integer.npy')
    pc_woo_db = (20 * np.log10(abs(pc_woo)))
    pc_woo_db_normalized = pc_woo_db - pc_woo_db.max()

    pc_chirp = np.load('data/pc_chirp_Win_integer.npy')
    pc_chirp_db = (20 * np.log10(abs(pc_chirp)))
    pc_chirp_db_normalized = pc_chirp_db - pc_chirp_db.max()

    pc_db = (20 * np.log10(abs(pc)))
    pc_db_normalized = pc_db - pc_db.max()
    plt.figure()
    plt.plot((distance), (pc_db_normalized), 'k*-')  # Matched Filter PC
    # plt.plot((pc_db_normalized), 'k*-')  # Matched Filter PC
    # plt.plot(fftshift(pc_db_normalized), 'k*-')
    # plt.plot( fftshift(pc_woo_db_normalized), 'k*-')  # Matched Filter PC
    # plt.plot( fftshift(pc_chirp_db_normalized), 'yo-')
    # plt.plot(fftshift(pc_woo_nowin_db_normalized), 'r^-')

    # plt.legend(['P4 code + window', 'Chirp + Window', 'P4 code without Window'])
    # plt.plot((distance), pc_dc_normalized,'k*-') # Matched Filter PC
    # plt.plot(fftshift(distance), fftshift(20 * np.log10(abs(pc))), 'k*-') # PC = mixer method
    # plt.xlabel('Sample Number')
    plt.xlabel('Distance [m]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Pulse Compression')
    plt.grid()
    #plt.xlim([-100, 1000])

    #'''

    fn.save_file_usrp(x_win_delay, len(x))
    plt.show()
