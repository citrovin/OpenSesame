import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout



# Function to extract features from audio file
def extract_features(file_name):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs

# Extract features from audio samples
features1 = extract_features("sample1.wav")
features2 = extract_features("sample2.wav")
features3 = extract_features("sample3.wav")


# Define the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(40,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
