import numpy as np
import matplotlib.pyplot as plt

def filter_signal(signal , sampling_freq , sampling_time,threshold):
    """
    :param signal: array containing signal values
    :param sampling_freq: the frequency at which data is sampled in khz
    :param sampling_time: the time interval in which data is sampled
    :return: filtered signal values in array
    """
    sampling_freq *= 1000
    signal = np.array(signal)
    dt = 1/sampling_freq
    time_interval = np.arange(0,sampling_time,dt)
    n = len(signal) # n discrete signal values
    fhat = np.fft.fft(signal)
    PSD = fhat * np.conjugate(fhat) / n
    freq_ar = (sampling_freq/n) * np.arange(n)
    threshold_array = PSD > threshold
    filtered_PSD = PSD * threshold_array
    filtered_fhat = fhat * threshold_array
    filtered_signal = np.fft.ifft(filtered_fhat)
    return filtered_signal

#example
# dt = .0001
# t = np.arange(0,2,dt)
# f1 = 2*np.pi*100 ; f2 = 2*np.pi*120
# signal = np.sin(f1*t) + np.sin(f2*t)
# clean_signal = signal
# signal += 2.5*np.random.randn(len(t))
#
#
#
# plt.plot(t,signal,color = 'r')
# plt.show()
