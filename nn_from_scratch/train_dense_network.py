# %%
import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, InputLayer
from tensorflow_addons.optimizers import AdamW

from sklearn.model_selection import train_test_split

import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


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
features1 = extract_features("./data/sample1.wav")
features2 = extract_features("./data/sample2.wav")
features3 = extract_features("./data/sample3.wav")
features4 = extract_features("./data/sample4.wav")

features1 = np.swapaxes(features1, 0, 1)
features2 = np.swapaxes(features2, 0, 1)
features3 = np.swapaxes(features3, 0, 1)
features4 = np.swapaxes(features4, 0, 1)

labels1 = np.zeros(469)
labels2 = np.zeros(469)
labels3 = np.ones(469)
labels4 = np.ones(469)

features = [features1, features2, features3, features4]
labels = [labels1, labels2, labels3, labels4]

for i, elem in enumerate(features):
    print(f'Feature {i} shape: {elem.shape}')
    print(f'Label {i} shape: {labels[i].shape}\n')

# features = np.asarray(features).astype('float32')
features = np.array(features)
labels = np.array(labels)

# %%
# Define the model
model = Sequential()
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="OpenSesame",
    entity="juzay_and_co",
    config = model.get_config(),
    name = "Dense 2 Hidden Layers (469 => 256 => 128 => 1) | 75 Epochs"
)

#model.summary()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.2, 
    random_state=42
    )


# %%

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=75, 
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint("models")]
    )

run.finish()