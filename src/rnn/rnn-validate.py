from tensorflow import keras
import librosa
import numpy as np

import os


# Function to extract features from audio file
def extract_features(file_name, sample_rate):

    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=sample_rate)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    return mfccs



if __name__== "__main__" :

    SAMPLE_RATE = 19100
    
    features1 = extract_features("./data/validation/sample1.wav", sample_rate=SAMPLE_RATE) # positive philipp
    features2 = extract_features("./data/validation/sample2.wav", sample_rate=SAMPLE_RATE) # negative philipp
    features3 = extract_features("./data/validation/sample3.wav", sample_rate=SAMPLE_RATE) # negative dalim
    features4 = extract_features("./data/validation/sample4.wav", sample_rate=SAMPLE_RATE) # postive dalim

    features1 = np.swapaxes(features1, 0, 1)
    features2 = np.swapaxes(features2, 0, 1)
    features3 = np.swapaxes(features3, 0, 1)
    features4 = np.swapaxes(features4, 0, 1)

    print(features1.shape)
    print(features2.shape)
    print(features3.shape)
    print(features4.shape)

    features = [features1,features2,features3,features4]

    model = keras.models.load_model('./rnn/models/rnn-srNone-epochs50-v1')

    outputs = [model.predict(np.expand_dims(i, axis=0)) for i in features]

    for i, out in enumerate(outputs):
        print(f"Output {i}: {np.mean(out)}")