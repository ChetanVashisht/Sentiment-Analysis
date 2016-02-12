# Sentiment analysis of audio conversations.

To run the program run main.py in the python terminal from Phase I folder.
	
	$ python main.py

Expected output

		>>> 0 035afDS.wav
			1 019auUF.wav
			...
			...
			...
			398 030anUN.wav 

			S 	H 	F 	A 	N
			Forest Classifier:
			[[9 0 2 0 9]
			 [0 9 0 3 1]
			 [5 0 8 1 1]
			 [0 5 1 9 2]
			 [4 2 0 0 7]]

			Accuracy: 55.8 %

The output is a confusion matrix of classifications of emotions `{Fear, Sadness, Anger, Neutral, Happiness}`.


Phase 1 of the project was to implement the sentiment analysis engine on a set of audio signals from the Italian Datsbase.
It detects the emotion of the person speaking based on the variation in frequency of the voice signal.
Phase 1 successfully implemented with upto 55% accuracy on a windows 8 machine and upto 50% on Ubuntu 14.04 machine.

Phase 2 was to deal with the speaker separation. It is a challenging task and requires the implementation of more features and much better research.
Simple MFCCs and pre emphasis will not so the trick. Do work on in the future.
Approches tried include removing chunks of the song bounded by consecutive zeros, if the peak of the chunk is lesser than a threshold value. The reconstruction from class.py is highly inaccurate. Reason:Unknown.
