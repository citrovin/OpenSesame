# %%
import os
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
EPOCHS = 40

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

name = f"dense-nn-sr{SAMPLE_RATE}-epochs{EPOCHS}-v7"
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

number_recordings=X_test.shape[0]
samples_per_rec = X_test.shape[1]

X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2]))

print(f'y test: {y_test}')
print(f'y test shape: {y_test.shape}')

#print(number_recordings)

result = model.predict(X_test)

result2 = np.array([])
for res in result:
    if res[0]>=0.3:
        result2 = np.append(result2, 1.0)
    else:
        result2 = np.append(result2, 0.0)

print(f'Result shape: {result.shape}')
print(f'Result2 shape: {len(result2)}')


# %%


from sklearn.metrics import classification_report
print('Classificaiton Report over all sample vectors:')
print(classification_report(result2, y_test))

# %%
#result_samples = np.array([np.mean(result[(i*197)+20:((i+1)*197)-20]) for i, k in enumerate(range(result.shape[0])) if k%samples_per_rec==0])
#true_samples = np.array([np.mean(y_test[i*197:(i+1)*197]) for i, k in enumerate(range(result.shape[0])) if k%samples_per_rec==0])



result_samples = [[]]*number_recordings
true_samples = [[]]*number_recordings
THRESHOLD = 0.5


i = 0
for k in range(result.shape[0]):
    if k%samples_per_rec==0:
        # print(i)
        result_samples[i] = np.mean(result[i*samples_per_rec:(i+1)*samples_per_rec])
        true_samples[i] = np.mean(y_test[i*samples_per_rec:(i+1)*samples_per_rec])

        i+=1

for i, res in enumerate(result_samples):
    if res>THRESHOLD:
        result_samples[i]=1.0
    else:
        result_samples[i]=0.0

y_pred = np.array(result_samples)
y_true = np.array(true_samples)

print('Classificaiton Report over all recordings:')
print(classification_report(y_pred, y_true))


# %%
