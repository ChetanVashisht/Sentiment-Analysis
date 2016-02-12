from numpy import argmax, sqrt, mean, diff, log
from scipy.signal import blackmanharris, fftconvolve
import scipy.io.wavfile as wav
from numpy.fft import rfft, irfft
# from parabolic import parabolic
from matplotlib.mlab import find

def zeroCrossing(signal, fs):
    # Find all indices right before a rising-edge zero crossing
    indices = find((signal[1:] >= 0) & (signal[:-1] < 0))
    return fs/mean(diff(indices))

def freq_from_fft(signal, fs):
    # Compute Fourier transform of windowed signal
    windowed = signal * blackmanharris(len(signal))
    f = rfft(windowed)
    i = argmax(abs(f))
    
    # Convert to equivalent frequency
    return fs*i/len(windowed)

def autocorrelation(sig, fs):
    # Find the auto correlation which is a convolution with a reversed signal.
    # Reject the second half to remove the lag coefficents.
    corr = fftconvolve(sig, sig[::-1], mode = 'full')
    corr = corr[len(corr)/2:]

    # Find occurance of a minimum. Need the array index.
    # It is unrelable due to noise. Hence we take the nect value.
    d = diff(corr)
    start = find(d > 0)[0]
    peak = argmax(corr[start:]) + start
    return fs/peak

def pitch(signal, fs):
    # Extracting all the pitch and fundamental frequencies and returning it.
    feat = []
    feat.append(zeroCrossing(signal, fs))
    feat.append(freq_from_fft(signal, fs))
    feat.append(autocorrelation(signal, fs))
    return feat
