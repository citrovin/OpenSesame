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

SAMPLE_RATE = None#int(22050)
OUTPUT_DIR = "./models"
EPOCHS = 50


features1_pos = extract_features("../data/train/sample1_pos.wav", sample_rate=SAMPLE_RATE)
features2_pos = extract_features("../data/train/sample2_pos.wav", sample_rate=SAMPLE_RATE)
features3_pos = extract_features("../data/train/sample3_pos.wav", sample_rate=SAMPLE_RATE)
features4_pos = extract_features("../data/train/sample4_pos.wav", sample_rate=SAMPLE_RATE)


features1_neg = extract_features("../data/train/sample1_neg.wav", sample_rate=SAMPLE_RATE)
features2_neg = extract_features("../data/train/sample2_neg.wav", sample_rate=SAMPLE_RATE)
features3_neg = extract_features("../data/train/sample3_neg.wav", sample_rate=SAMPLE_RATE)
features4_neg = extract_features("../data/train/sample4_neg.wav", sample_rate=SAMPLE_RATE)
features5_neg = extract_features("../data/train/sample1_pos_neg.wav", sample_rate=SAMPLE_RATE)



features1_pos = np.swapaxes(features1_pos, 0, 1)
features2_pos = np.swapaxes(features2_pos, 0, 1)
features3_pos = np.swapaxes(features3_pos, 0, 1)
features4_pos = np.swapaxes(features4_pos, 0, 1)

features1_neg = np.swapaxes(features1_neg, 0, 1)
features2_neg = np.swapaxes(features2_neg, 0, 1)
features3_neg = np.swapaxes(features3_neg, 0, 1)
features4_neg = np.swapaxes(features4_neg, 0, 1)
features5_neg = np.swapaxes(features5_neg, 0, 1)

# labels1_pos = np.ones(features1_pos.shape[0])
# labels2_pos = np.ones(features1_pos.shape[0])
# labels3_pos = np.ones(features1_pos.shape[0])
# labels4_pos = np.ones(features1_pos.shape[0])

# labels1_neg = np.zeros(features1_pos.shape[0])
# labels2_neg = np.zeros(features1_pos.shape[0])
# labels3_neg = np.zeros(features1_pos.shape[0])
# labels4_neg = np.zeros(features1_pos.shape[0])
# labels5_neg = np.zeros(features1_pos.shape[0])

labels1_pos = np.ones(1)
labels2_pos = np.ones(1)
labels3_pos = np.ones(1)
labels4_pos = np.ones(1)

labels1_neg = np.zeros(1)
labels2_neg = np.zeros(1)
labels3_neg = np.zeros(1)
labels4_neg = np.zeros(1)
labels5_neg = np.zeros(1)



features = np.array([
    features1_pos,
    features2_pos,
    features3_pos,
    features4_pos,
    #features1_neg,
    features2_neg,
    features3_neg,
    features4_neg,
    #features5_neg
    ])

labels = np.array([
    np.expand_dims(labels1_pos, axis=1),
    np.expand_dims(labels2_pos, axis=1),
    np.expand_dims(labels3_pos, axis=1),
    np.expand_dims(labels4_pos, axis=1),
    #np.expand_dims(labels1_neg axis=1),
    np.expand_dims(labels2_neg, axis=1),
    np.expand_dims(labels3_neg, axis=1),
    np.expand_dims(labels4_neg, axis=1),
    #np.expand_dims(labels5_neg axis=1),
    ])

print(features.shape)
print(labels.shape)

# for i, elem in enumerate(features):
#     print(f"Feature {i+1} shape: {elem.shape}")
#     print(f"Label {i+1} shape: {labels[i].shape}\n")

# %%
# Define the model
model = Sequential()
model.add(InputLayer((187,40)))
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
    epochs=EPOCHS,
    batch_size=1,
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

name = f"rnn-sr{SAMPLE_RATE}-epochs{EPOCHS}-v1"
model.save(os.path.join(OUTPUT_DIR, name))