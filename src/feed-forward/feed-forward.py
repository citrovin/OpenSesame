# %%
import os
import librosa
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, InputLayer

from sklearn.model_selection import train_test_split

import tensorflow as tf

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from matplotlib import pyplot as plt

import sys
sys.path.append('../../src')
from utils.preprocess_data import loadData


# %%
# Extract features from audio samples

SAMPLE_RATE = 48000#None#int(16000)
OUTPUT_DIR = "./models"
EPOCHS = 2

# Import data
x_train, y_train, x_test, y_test = loadData(asTensor=False)


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
#     name = "Feed-Forward"
# )

# Split the data into training and test sets
X, y, _, _ = loadData(asTensor=False)
X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# %%

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    validation_data=(X_test, y_test)
)

name = 'test'#f"dense-nn-sr{SAMPLE_RATE}-epochs{EPOCHS}-v5"
# Plot accuarcy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig(f'./plots/{name}_acc.png')


# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
plt.savefig(f'./plots/{name}_loss.png')

# callbacks=[
#     WandbMetricsLogger(log_freq=5),
#     WandbModelCheckpoint("models")]


# wandb.finish()

model.save(os.path.join(OUTPUT_DIR, name))

# %%

_, _, X_test, y_test = loadData(asTensor=False)
print(X_test.shape)
number_recordings=X_test.shape[0]
X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
print(X_test.shape)
print(y_test.shape)
print(y_test)
print(number_recordings)

result = model.predict(X_test)

#print(result)
print(result.shape)

#%%
result_samples = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
true_samples = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
i = 0
for k in range(result.shape[0]):
    if k%197==0:
        # print(i)
        # result_samples[i] = (result[i*197:(i+1)*197])
        #result_samples[i] = np.mean(result[(i*197)+20:((i+1)*197)-20])
        result_samples[i] = np.mean(result[(i*197)+20:((i+1)*197)-20])

        result_samples[i] = 1.0 if result_samples[i] > 0.3 else 0.0
        # true_samples[i] = (y_test[i*197:(i+1)*197])
        true_samples[i] = np.mean(y_test[i*197:(i+1)*197])

        i+=1


#print(len(result_samples))
#print(len(true_samples))
y_pred = np.array(result_samples)
y_true = np.array(true_samples)
print(y_pred)
print(y_true)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_true))

# %%
