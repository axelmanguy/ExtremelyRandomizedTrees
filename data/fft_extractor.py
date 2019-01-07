import glob
import os
import scipy.io.wavfile
import numpy as np
genres = ['classical', 
		  'disco',
		  'rock',
		  'hiphop',
		  'jazz',
		  'reggae'
		  'pop',
		  'blues',
		  'metal',
		  'country']
X,Y=[],[]
for i,genre in enumerate(genres):
	for j,filename in enumerate(glob.glob(os.path.join('./', genre, "*.wav"))):
		sample_rate, data = scipy.io.wavfile.read(filename)
		fft_features = abs(scipy.fft(data)[:2000])
		X.append(fft_features)
		Y.append(genre)
		print(genre+" : "str(j)+"%")
print(np.shape(X))
print(np.shape(Y))
np.save('X', X)
np.save('Y', Y)