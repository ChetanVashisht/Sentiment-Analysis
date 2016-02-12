#!/usr/bin/env python
import numpy as np
import scipy.io.wavfile as wav
from MFCC import *
from BandpassFilters import *
from PitchDetection import *
from os import listdir
import csv

def featureExtraction(string):
        # Extract the features of each file and store it in a csv file.
        plot = 0
        size1 = 256
        f_min = 50
        f_max = 16000
        N = 26
        sampling_interval = 400
        frame_size = 160

        # Loading the bandpass filters from a csv file which I wrote earlier, containing 25 triangular filters fitted between 0 and 16000Hz normalised in the Mel scale.
        filters = np.genfromtxt("Bandpass_filters.csv", delimiter = ',')

        # Loading a song from our database to find it's MFCCs and defining the rate, frame size 
        (rate, sig) = wav.read("Dataset/" + string)

        # Modify the signal and get the MFCCs and the min, max and avg energies.
        sig = modify_signal(sig, sampling_interval, frame_size)
        MFCC, DDFC, energy = features(sig, sampling_interval, frame_size, size1, plot = 0)
        pitches = pitch(sig, rate)
        # The frature extracted vector with the filename and label
        feat = []
        # feat = zip(string, MFCC, DDFC, energy, string[-5])
        feat.append(string)
        for i in MFCC:
                feat.append(i)
        for i in DDFC:
                feat.append(i)
        for i in energy:
                feat.append(i)
        for i in pitches:
                feat.append(i)
        feat.append(string[-5])
        return (feat)

def feat_to_csv():
	# Defining the constants and generating the bandpass filters.
	# Open the stored features from the csv file and 
	plot = 0
	size1 = 256
	f_min = 50
	f_max = 16000
	N = 26
	sampling_interval = 400
	frame_size = 160
	generate_bandpass_filters(f_min, f_max, N, size1, plot = 0)

	file_names = listdir('Dataset')
	f = open( 'Features.csv' , 'wb' )
	b = csv.writer( f, delimiter = ',')

	# Writing the csv file
	count = 0
	for file_name in file_names:
		print count, file_name
		feat = featureExtraction(file_name)
		count += 1
		b.writerow(feat)

	f.close()
