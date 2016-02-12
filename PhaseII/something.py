from numpy import concatenate, delete, vstack, zeros, savetxt, genfromtxt
from sklearn import mixture
from scipy.io.wavfile import read, write
from MFCC import frame_fft, mfcc

rate, signal1 = read("conv.wav")
signal = signal1
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
savetxt("MFCC.csv", MFCC, delimiter = ',')
MFCC = genfromtxt("MFCC.csv", delimiter = ',')
# clf = mixture.GMM(n_components = [0,1], covariance_type = 'full')
# clf.fit(MFCC)
# prob = clf.score_samples(MFCC)

print MFCC.shape
