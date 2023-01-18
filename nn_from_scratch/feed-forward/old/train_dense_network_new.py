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

sample_rate = None#int(22050)


features1 = extract_features("./data/sample1.wav", sample_rate=sample_rate)
features2 = extract_features("./data/sample2.wav", sample_rate=sample_rate)
features3 = extract_features("./data/sample3.wav", sample_rate=sample_rate)
features4 = extract_features("./data/sample4.wav", sample_rate=sample_rate)



features1 = np.swapaxes(features1, 0, 1)
features2 = np.swapaxes(features2, 0, 1)
features3 = np.swapaxes(features3, 0, 1)
features4 = np.swapaxes(features4, 0, 1)

print(features1.shape)


labels1 = np.zeros(features1.shape[0])
labels2 = np.zeros(features1.shape[0])
labels3 = np.ones(features1.shape[0])
labels4 = np.ones(features1.shape[0])

print(labels1.shape)

# features = [features1, features2, features3, features4]
# labels = [labels1, labels2, labels3, labels4]

features = np.vstack([features1, features2, features3, features4])
labels = np.hstack([labels1, labels2, labels3, labels4])


print(f'Feature shape: {features.shape}')
print(f'Label shape: {labels.shape}')

# %%
# Define the model
model = Sequential()
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
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

#model.summary()'''

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.2, 
    random_state=42
    )

print(f"x_train: {X_train.shape}")
print(f"x_train: {X_test.shape}")


# %%

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    validation_data=(X_test, y_test)
    
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plt.savefig('./loss_acc.png')

# callbacks=[
#     WandbMetricsLogger(log_freq=5),
#     WandbModelCheckpoint("models")]


# wandb.finish()