# %%
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm



# %%
# Function to extract features from audio file
def extract_features(file_name):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs

# %%
# Extract features from audio samples
features1 = extract_features("/Users/dalimwahby/Documents/EIT/UCA/Data_Science_for_Business/nn/data/sample1.wav")
features2 = extract_features("/Users/dalimwahby/Documents/EIT/UCA/Data_Science_for_Business/nn/data/sample2.wav")
features3 = extract_features("/Users/dalimwahby/Documents/EIT/UCA/Data_Science_for_Business/nn/data/sample3.wav")
features4 = extract_features("/Users/dalimwahby/Documents/EIT/UCA/Data_Science_for_Business/nn/data/sample4.wav")

# labels1 = np.zeros(1)
# labels2 = np.zeros(1)
# labels3 = np.ones(1)
# labels4 = np.ones(1)

features1 = np.swapaxes(features1, 0, 1)
features2 = np.swapaxes(features2, 0, 1)
features3 = np.swapaxes(features3, 0, 1)
features4 = np.swapaxes(features4, 0, 1)

labels1 = np.zeros(469)
labels2 = np.zeros(469)
labels3 = np.ones(469)
labels4 = np.ones(469)

# tensor1 = [1, ]

features = [features1, features2, features3, features4]
labels = [labels1, labels2, labels3, labels4]

for i, elem in enumerate(features):
    print(f'Feature {i} shape: {elem.shape}')
    print(f'Label {i} shape: {labels[i].shape}\n')

# features = np.asarray(features).astype('float32')
features = np.array(features)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# %%
# Define the model
model = svm.SVC()
model.fit(X_train, y_train)
