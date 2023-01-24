# %%
import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, InputLayer

from sklearn.model_selection import train_test_split

import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from matplotlib import pyplot as plt

import os

import sys
sys.path.append('../../src')
from utils.preprocess_data import loadData


# %%
# Function to extract features from audio file
def extract_features(file_name, sample_rate):
    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=sample_rate)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs

# %%
# Extract features from audio samples

# RNN
x_train, y_train, x_test, y_test = loadData(asTensor=True)

SAMPLE_RATE = None#int(22050)
OUTPUT_DIR = "./models"
EPOCHS = 50


# %%
# Define the model
model = Sequential()
model.add(InputLayer((197,40)))
model.add(SimpleRNN(20))
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="OpenSesame",
#     entity="juzay_and_co",
#     config = model.get_config(),
#     name = "Dense 2 Hidden Layers (469 => 256 => 128 => 1) | 75 Epochs | Sample rate: sample_rate=int(22050/4)"
# )


# %%

# Train the model
history = model.fit(
    x_train, 
    y_train, 
    epochs=EPOCHS,
    batch_size=1,
    validation_data=(x_test, y_test)
    
)


name = f"rnn-sr{SAMPLE_RATE}-epochs{EPOCHS}-v1"
# Plot the Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plt.savefig(f'./plots/{name}_loss.png')

# Plot the Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plt.savefig(f'./plots/{name}_loss.png')

# callbacks=[
#     WandbMetricsLogger(log_freq=5),
#     WandbModelCheckpoint("models")]


# wandb.finish()

model.save(os.path.join(OUTPUT_DIR, name))
