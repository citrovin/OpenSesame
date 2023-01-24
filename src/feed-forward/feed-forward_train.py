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

name = f"dense-nn-sr{SAMPLE_RATE}-epochs{EPOCHS}-v8"
# Plot accuarcy

fig, ax = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(12, 5)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')
#plt.show()
#plt.savefig(f'./plots/{name}_acc.png')


# Plot Loss
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')

fig.savefig(f'./plots/plots_{name}.png')

plt.close()

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


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print('Classificaiton Report over all sample vectors:')
print(classification_report(result2, y_test))

labels_cf_matrix = ['True Neg','False Pos','False Neg','True Pos']
labels_cf_matrix = np.asarray(labels_cf_matrix).reshape(2,2)

cf_matrix = confusion_matrix(y_test, result2)
heatmap_sns = sns.heatmap(cf_matrix, annot=labels_cf_matrix, cmap='Blues', fmt='')
heatmap_sns.get_figure().savefig(f'./plots/heatmap_samples_{name}.png')

plt.close()
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

cf_rec_matrix = confusion_matrix(y_true, y_pred)
heatmap_rec_sns = sns.heatmap(cf_matrix, annot=labels_cf_matrix, cmap='Blues', fmt='')
heatmap_rec_sns.get_figure().savefig(f'./plots/heatmap_recordings_{name}.png')


# %%
