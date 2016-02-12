from scipy.io.wavfile import read, write
from numpy import concatenate, delete, vstack, zeros, savetxt
from MFCC import frame_fft, mfcc

class part:
	def __init__(self, start, signal):
		self.start = start
		self.maximum, self.end = self.locator(signal[start:], start)
		self.maxima = abs(signal[self.maximum])
		
	def sign(self, i):
		# If it is positive return True, else return False.
		if i > 0:
			return True
		return False
	
	def locator(self, signal, start):
		# Look for the general direction, then locate the maxima and zero value.
		maximum, positive = 0, False
		i = 0
		while(signal[i] == 0):
			i += 1
		
		positive = self.sign(signal[i])
			
		while (self.sign(signal[i]) == positive and i + start < length):
			i += 1
			# print signal[i]
			if(positive == True):
				if(signal[i] > signal[maximum]):
					maximum = i
			else:
				if(signal[i] < signal[maximum]):
					maximum = i
		
		maximum = start + maximum
		end = start + i
		return maximum, end
	
	def dispaly(self):
		# Prints the elements of the class
		print "\nStart " + str(self.start) + "\tMaximum " + str(self.maximum) + "\tEnd " + str(self.end)

rate, signal = read("conv.wav")
simplelist = []
i = 0
length = len(signal)
while i < len(signal)-100:
    # print i
    object = part(i, signal)
    i = object.end 
    simplelist.append(object)
print "end"
global_max = max( map( lambda x: x.maxima, simplelist))
new_list = filter( lambda x: x.maxima >= float(global_max)/16, simplelist)

print len(new_list), len(simplelist)

sig = []
for i in new_list:
    sig = concatenate([sig, signal[i.start:i.end]])

# write("edited.wav", rate, sig)
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

