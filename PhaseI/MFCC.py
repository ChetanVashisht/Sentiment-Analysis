import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import dct
import csv as csv
import matplotlib.pyplot as plt

# Defining helper functions
def modify_signal(sig, sampling_interval, frame_size):
	# Modifies the file to make it fit sampling rate and frame size
	b = (len(sig) - sampling_interval) % frame_size
	sig = sig[:-b]
	return sig

def freq_to_mel(f):
	# Convert to the Mel scale
	return 1125*np.log(1 + f/700.0)

def mel_to_freq(m):
	# Convert Mel to normal frequency scale
	return 700*(np.exp(m/1125.0) - 1)

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

def frame_power(S, N):
	# Calculating the absolute power of the frame
	return 1/N * (np.sum(np.abs(S))**2)

def mfcc(signal_fft, plot = 0, no_of_coeff = 13):
        # Mel Cepstal Frequency Coefficents
        # frame -> filters -> log -> DCT -> return first 13 coeff
        filters = np.genfromtxt("Bandpass_filters.csv", delimiter = ',')
        energy = []
        signal_fft = np.multiply(signal_fft, signal_fft)
        for i in range(len(filters)):
            energy.append(sum(abs(signal_fft)*filters[i]))

        if (np.sum(energy) != 0):
            energy = np.log10(energy)
            energy = dct(energy)
        if(plot == 1):
            plt.plot(energy[:no_of_coeff])
            plt.show()
        return energy[:no_of_coeff]

def ddfc(MFCC, no_of_coeff = 13):
    # Delta Delta Frequrncy Coefficients
    DDFC = np.zeros(no_of_coeff)
    for i, mfcc in enumerate(MFCC):
        if (i==0 or i==1 or i == len(MFCC)-2 or i == len(MFCC)-1):
            continue
        else:
            DFCC = np.vstack([DDFC,(2*(MFCC[i+2] - MFCC[i-2]) + (MFCC[i+1] - MFCC[i-1]))/10])
                    
    return np.mean(DFCC, axis = 0)

def features(signal, sampling_interval = 400, frame_size = 160, size1 = 256, plot = 0):
	i = 0
	signal_fft = np.zeros(size1)
	frame_energy = []
	no_of_coeff = 13
	# Get the array of framed signals and the power of the frames
	while(i + sampling_interval < len(signal)):
		frame = frame_fft(signal[i:i+sampling_interval], window = 'Hamming', K = 512, N = sampling_interval, size1 = size1)
		signal_fft = np.vstack([signal_fft, frame])
		frame_energy.append(frame_power(frame, float(sampling_interval)))
		i = i + frame_size

	signal_fft = np.delete(signal_fft, (0), axis = 0)
	MFCC = np.zeros(no_of_coeff)
	for i in signal_fft:
		MFCC = np.vstack([MFCC, mfcc(i)])

	MFCC = np.delete( MFCC, 0, axis = 0)
	DDFC = ddfc(MFCC, no_of_coeff)

	# Store the max, min and the mean energy of the signal in feat.
	feat = []
	feat.append(max(frame_energy))
	feat.append(min(frame_energy))
	feat.append(sum(frame_energy)/len(frame_energy))

	# Return the first 13 MFCC coefficients and feat (which contains the minimum, maximum and average energies of the frames)
	return np.mean(MFCC, axis = 0), DDFC, feat
