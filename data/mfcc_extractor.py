import glob
import os
import scipy.io.wavfile
import numpy as np
from python_speech_features import mfcc
genres=[  'classical',
		  'disco',
		  'rock',
		  'hiphop',
		  'jazz',
		  'blues',
		  'metal',
		  'country']
X,Y=[],[]
for i,genre in enumerate(genres):
	for j,filename in enumerate(glob.glob(os.path.join('./', genre, "*.wav"))):
		rate, data = scipy.io.wavfile.read(filename)
		mfcc_feat = mfcc(data,rate,nfft=1024)
		mean_features = np.mean(mfcc_feat, axis=0)
		print(np.shape(mean_features))
		X.append(mean_features)
		print(np.shape(X))
		Y.append(genre)
		print(genre+" : "+str(j)+"%")
print(np.shape(X))
print(np.shape(Y))
np.save('X_MFCC', X)
np.save('Y_MFCC', Y)