#%%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import librosa
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing

import sys
from importlib import reload # python 2.7 does not require this

sys.path.append('../../src')
from utils.preprocess_data import loadData

import pickle

#%%
def calculate_delta(array):
   
    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined



#%%
def extract_mfcc(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    mfccs = np.swapaxes(mfccs,0,1)
    return mfccs



# %%
X, y, _, _ = loadData(asTensor=False)
print(X.shape)
X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
print(X.shape)
print(y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

clf = SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_val, y_val)
print("Accuracy: ", accuracy)
s = pickle.dumps(clf)
with open(r"models/svm.pickle", "wb") as output_file:
    pickle.dump(clf, output_file)
#%%
_, _, X_test, y_test = loadData(asTensor=False)

result = clf.predict(X_test)
print(result.shape)

result_samples = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
true_samples = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
i = 0
for k in range(result.shape[0]):
    if k%197==0:
        # print(i)
        # result_samples[i] = (result[i*197:(i+1)*197])
        result_samples[i] = np.mean(result[(i*197)+0:((i+1)*197)-0])
        result_samples[i] = 1.0 if result_samples[i] > 0.15 else 0.0
        # true_samples[i] = (y_test[i*197:(i+1)*197])
        true_samples[i] = np.mean(y_test[i*197:(i+1)*197])

        i+=1
print(len(result_samples))
print(len(true_samples))
y_pred = np.array(result_samples)
y_true = np.array(true_samples)
print(y_pred)
print(y_true)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_true)
# accuracy_score(result, y_test)
# score = clf.score(result, y_test)


# %%

sr,audio = read(file_path+'pos'+".wav")
vector = extract_features(audio,sr)

result = clf.predict(vector)

print(result)
# %%
sr,audio = read(file_path+'output'+".wav")
vector = extract_features(audio,sr)

result = clf.predict(vector)

print(result)

# %%
