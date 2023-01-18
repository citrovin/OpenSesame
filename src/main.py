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

def read_files(dir_path):
    file_paths = []
    # iterate over files in
    # that directory
    for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            file_paths.append(f)

    return file_paths


if __name__== "__main__" :

    SAMPLE_RATE = None

    file_paths = read_files('./data/validation/')

    '''features1 = extract_features("./data/validation/sample1.wav", sample_rate=SAMPLE_RATE) # positive philipp
    features2 = extract_features("./data/validation/sample2.wav", sample_rate=SAMPLE_RATE) # negative philipp
    features3 = extract_features("./data/validation/sample3.wav", sample_rate=SAMPLE_RATE) # negative dalim
    features4 = extract_features("./data/validation/sample4.wav", sample_rate=SAMPLE_RATE) # postive dalim

    features1 = np.swapaxes(features1, 0, 1)
    features2 = np.swapaxes(features2, 0, 1)
    features3 = np.swapaxes(features3, 0, 1)
    features4 = np.swapaxes(features4, 0, 1)

    features = [features1,features2,features3,features4]

    model = keras.models.load_model('./feed-forward/models/dense-nn-srNone-epochs100-v3')

    output1 = model.predict(features1)
    output2 = model.predict(features2)
    output3 = model.predict(features3)
    output4 = model.predict(features4)

    print(f"Output 1: {np.mean(output1)}")
    print(f"Output 2: {np.mean(output2)}")
    print(f"Output 3: {np.mean(output3)}")
    print(f"Output 4: {np.mean(output4)}")
'''

    #print(model.summary())