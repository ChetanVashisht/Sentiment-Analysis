import numpy as np
import matplotlib.pyplot as plt

def freq_to_mel(f):
	# Convert to the Mel scale
	return 1125*np.log(1 + f/700.0)

def mel_to_freq(m):
	# Convert Mel to normal frequency scale
	return 700*(np.exp(m/1125.0) - 1)

def f_bin(h, rate = 16000, fft_size = 256):
	# Since we can't get the exact required frequencies, we have to approximate them. 
	f = np.floor(h*fft_size/rate)
	return f

def triangle_filter(f1, f2, f3, fmax):
	# Creates a triangular filter with f(n-1), f(n), f(n+1)
	h = []
	for f in range(fmax):
		if(f < f1):
			h.append(0)
		elif(f1 <= f <= f2):
			h.append((f - f1)/(f2 - f1))
		elif(f2 <= f <= f3):
			h.append((f3 - f)/(f3 - f2))
		else:
			h.append(0)
	return h

def generate_bandpass_filters(f_min = 50, f_max = 16000, N = 26, size = 256, plot = 0):
	# Convert to the mel scale and divide the interval into 'N' parts. Then convert it back to normal frequencies. This gives us the mid points of all the band pass filters. To create the filter, join the (n-1)th frequency point to the nth prequency point to the (n+1)th frequency point. (n-1)th and (n+1)th are the lower ends of the triangle filters and nth is the mid point.
	f_min1 = freq_to_mel(f_min)
	f_max1 = freq_to_mel(f_max)
	f = []
	f = np.linspace(f_min1, f_max1, N)

	f = mel_to_freq(f)
	# f now contains the converted mel frequencies [300, 390,... 16000]
	# f contains the bins of importance [9, 16, ... 255]
	f = f_bin(f)

	g = np.zeros((N - 2, size))
	for i in range(N-2):
		g[i] = triangle_filter(f[i], f[i+1], f[i+2], size)
	g = np.array(g)
	
	# Plot the bandpass filters if required.
	if(plot == 1):
		plt.plot(np.arange(0,size1),filters[:,:].T)
		plt.show()

	# Store the numpy array into a csv file for quick access.
	# There is no need to compute the bandpass filters for each song.
	np.savetxt("Bandpass_filters.csv", g, delimiter = ',')