# %%
import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


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
features1 = extract_features("/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/sample1.wav")
features2 = extract_features("/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/sample2.wav")
features3 = extract_features("/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/sample3.wav")
features4 = extract_features("/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/sample4.wav")

labels1 = np.zeros(469)
labels2 = np.zeros(469)
labels3 = np.ones(469)
labels4 = np.ones(469)

features1 = np.swapaxes(features1, 0, 1)
features1 = np.swapaxes(features2, 0, 1)
features1 = np.swapaxes(features3, 0, 1)
features1 = np.swapaxes(features4, 0, 1)

# labels1 = np.zeros(40)
# labels2 = np.zeros(40)
# labels3 = np.ones(40)
# labels4 = np.ones(40)

# tensor1 = [1, ]
print(features1.shape)


features = [features1, features2, features3, features4]
labels = [labels1, labels2, labels3, labels4]
# features = np.asarray(features).astype('float32')
features = np.array(features)
labels = np.array(labels)

print(features.shape)
print(labels.shape)

# %%
# Define the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(40,469)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print(features1)
# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


# SVM
# %%
