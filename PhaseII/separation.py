import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import dct
from math import sqrt
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------
def mfcc(signal_fft, plot = 0, no_of_coeff = 13):
    # Mel Cepstal Frequency Coefficents
    # frame -> filters -> log -> DCT -> return first 13 coeff
    filters = np.genfromtxt("Bandpass_filters.csv", delimiter = ',')
    energy = []
    for i in range(len(filters)):
            energy.append(sum(abs(signal_fft)*filters[i]))

    if (np.sum(energy) != 0):
            energy = np.log10(energy)
            energy = dct(energy)
    if(plot == 1):
            plt.plot(energy[:no_of_coeff])
            plt.show()
    # print energy[:13]
    return energy[:no_of_coeff]

def frame_fft(signal, window = 'Hamming', K = 512, N = 400, size1 = 256):
    # Takes the DFT of the frame in consideration. K is the fft size and N is the length of the window. signal now contains the windowed function.
    h = []
    s1 = []
    if(window == 'Hamming'):
            for n in range(N):
                    h = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
                    signal[n] = signal[n]*h

    elif(window == 'Blackman'):
            for n in range(N):
                    h = 0.42 - 0.49*np.cos(2*np.pi*n/(N-1)) + 0.07*np.cos(4*np.pi*n/(N-1))
                    signal[n] = signal[n]*h

    elif(window == 'Blackman-Harris'):
            for n in range(N):
                    h = 0.35 - 0.49*np.cos(2*np.pi*n/(N-1)) + 0.14*np.cos(4*np.pi*n/(N-1)) - 0.01*np.cos(6*np.pi*n/(N-1))
                    signal[n] = signal[n]*h

    # Taking the fft of the framed signal
    S = fft(signal, K)

    # fft give transform from -f to +f. So we return only [0,f].
    return np.array(S[:size1])

#--------------------------------------------------------------------------
def distance(feat1, feat2):
    dist = 0
    for i in xrange(len(feat1)):
        dist += (feat1[i]-feat2[i])**2
    dist = sqrt(dist)
    return dist

#---------------------------------------------------------------------------
#(rate, signal) = wav.read("004abUA.wav")
#(rate, signal) = wav.read("002acDF.wav")
(rate, signal1) = wav.read("conv.wav")
signal = signal1[:800000]
signal_fft = np.zeros(256)
i = 0
while(i + 400 < len(signal)):
    signal_fft = np.vstack([signal_fft, frame_fft(signal[i:i+400])])
    i += 160

signal_fft = np.delete(signal_fft, 0, axis = 0)
MFCC = np.zeros(13)
for i in signal_fft:
    temp = mfcc(i)
    MFCC = np.vstack([MFCC, temp])

MFCC = np.delete(MFCC, 0, axis = 0)

#--------------------------------------------------------------------------
maximum = np.amax(MFCC, axis = 0)
minimum = np.amin(MFCC, axis = 0)

MFCC = (MFCC - minimum)/(maximum - minimum)
# print maximum, minimum
weights = np.var(MFCC, axis = 0)

Weighted_MFCC = np.multiply(MFCC, weights)
Weighted_MFCC = np.sum(Weighted_MFCC, axis = 1)
print Weighted_MFCC, Weighted_MFCC.shape

#--------------------------------------------------------------------------
plt.plot(xrange(len(Weighted_MFCC)), Weighted_MFCC)
plt.axhline(y = np.mean(Weighted_MFCC))
plt.axis([0, len(Weighted_MFCC), 0, 0.35])
plt.show()

#--------------------------------------------------------------------------
