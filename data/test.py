import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import glob
import os
import scipy.io.wavfile
import numpy as np
"""
def plot_specgram(ax, fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ax.specgram(X, Fs=sample_rate, xextent=(0, 30), cmap='jet')

def plot_spectrum(ax, fn):
    rate, X = scipy.io.wavfile.read(fn)
    spectrum = np.fft.fft(X)
    freq = np.fft.fftfreq(len(X), 1.0 / rate)
    #ax.specgram(X, Fs=sample_rate, xextent=(0, 30), cmap='jet')
    ax.plot(freq, abs(spectrum))
    ax.set_xlim(0,2000)

plt.clf()
genres = ["classical", "jazz", "country", "pop", "rock", "metal"]
num_files = 3
f, axes = plt.subplots(len(genres), num_files)
for genre_idx, genre in enumerate(genres):
    for idx, fn in enumerate(glob.glob(os.path.join('./', genre, "*.wav"))):
        if idx == num_files:
            break
        axis = axes[genre_idx, idx]
        axis.yaxis.set_major_formatter(EngFormatter())
        axis.set_title("%s song %i" % (genre, idx + 1))
        plot_spectrum(axis, fn)

specgram_file = os.path.join('./', "Spectrogram_Genres.png")
plt.savefig(specgram_file, bbox_inches="tight")

plt.show()
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X=np.load('X.npy')
Y=np.load('Y.npy')
print(np.shape(X))
print(np.shape(Y))
score = []
n_iter=10
for i in range(n_iter):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    classifier = ExtraTreesClassifier(n_estimators=200, criterion='entropy')
    classifier.fit(X_train,y_train);
    score += [classifier.score(X_test,y_test)]
    y_pred = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test,y_pred)
    plt.imshow(cnf_matrix,cmap=plt.cm.Blues)
    print(np.unique(y_pred))
    print('*', end='')
print(" done!")
print(score)
plt.show()

print("Average generalization score:", np.mean(score))
print("Standard deviation:", np.std(score))