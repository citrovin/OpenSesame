#%%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import librosa
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
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


#%%
file_path = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/"

files = ["sample1", "sample2","sample3", "sample4"]

features = np.asarray(())
for file in files:
    # audio, sample_rate = librosa.load(file_path+file+".wav", sr=None)
    sr,audio = read(file_path+file+".wav")
    vector = extract_features(audio,sr)
    # vector = extract_features(audio,sample_rate)
    print(vector.shape)
    # vector = extract_mfcc(file_path+file+".wav")
    print(features.shape)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

# labels1 = np.zeros(215)
# labels2 = np.zeros(215)
# labels3 = np.ones(215)
# labels4 = np.ones(215)

len_samples = 498
labels = np.hstack([np.zeros(len_samples),np.zeros(len_samples),np.ones(len_samples),np.ones(len_samples)])


#%%
print(features.shape)
print(labels.shape)

# %%

# Make sure to extract mfcc features from each audio file before running this code
X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)
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
