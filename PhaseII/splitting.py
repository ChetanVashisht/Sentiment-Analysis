from numpy import 
from os import listdir
from scipy.io.wavfile import read, write
from MFCC import frame_fft, mfcc

# data = csvread('MFCC.csv')
# [IDX, C] = kmeans(data, 2, 'diatance', 'sqEuclidean')
# csvwrite('IDX.csv', IDX)

def inputs(signal):
	signal_fft = zeros(256)
	i = 0
	while(i + 400 < len(signal)):
		signal_fft = vstack([signal_fft, frame_fft(signal[i : i+400], window = 'Hamming', K = 512, N = 400, size1 = 256)])
		i += 160
	signal_fft = delete(signal_fft, 0, axis = 0)

	MFCC = zeros(13)
	for i in signal_fft:
		MFCC = vstack([MFCC, mfcc(i)])
	MFCC = delete( MFCC, 0, axis = 0)
	
	return MFCC

def labeller(MFCC, label):
	# For adding the label try hstack
	labels = label*ones(len(MFCC, axis = 1))
	MFCC = hstack(MFCC, labels)