from tensorflow import keras

import librosa
import numpy as np
import pyaudio
import wave

import time
import os
from utils.preprocess_data import loadData, extract_features

# call model
def speaker_identification(
                model, #specify
                name,
                buffer,
                sample_rate
                ):
    # save recorded files
    path_live_data = save_wav(name,buffer)
    
    # extract the features and create vectors
    ################# RNN #################
    features = extract_features(path_live_data, sample_rate=sample_rate) # return a np.array with the features
    #print(features.shape)
    features = np.swapaxes(features, 0, 1) # swap axis of features
    #print(features.shape)
    
    # predict the speaker or not
    output = model.predict(np.expand_dims(features, axis=0))
    
    # Choose if it is correct speaker
    
    return output

# save the recorded data in 2 files, asynchronously and call the model on the saved file
def save_wav(name,buffer):
    path_live_data = '../data/live/'+name+'.wav'
    with wave.open(path_live_data, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"".join(buffer))
            
    return path_live_data


# Function to extract features from audio file
def extract_features(file_name, sample_rate):

    # Load the audio file
    audio, sample_rate = librosa.load(file_name, sr=sample_rate) # retruns array of float values form file
    print(f'Shape read from load: {audio.shape}') # nd_array
    #print(sample_rate) # SAMPLE_RATE

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    print(f'Shape of MFCCs after feature.mfcc: {mfccs.shape}')
    
    # Normalize the MFCCs
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    print(f'Shape after normalizing: {mfccs.shape}')

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

    
    # Constants. You can decrease the chunk if you want a faster loop (faster sample rate)
    CHUNK = 2048 # in buffer always 2 times the number of chunk is saved
    RECORDING_SECONDS = 2
    SAMPLE_RATE = 38000
    
    MODEL_PATH = './rnn/models/rnn-srNone-epochs50-v1'
    
    model = keras.models.load_model(MODEL_PATH)
    print('Model loaded')
    
    print("Opening stream..")
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK )
    stream.start_stream()


    buffer1=[]
    buffer2=[]
    frame_num = int( ((SAMPLE_RATE / CHUNK) * RECORDING_SECONDS))
    #print(frame_num)

    i = 0
    #print("Let's go! Good job Filippo!")
    while(1):
        i+=1
        data = stream.read(CHUNK)
        buffer1.append(data)
        buffer2.append(data)
        print(f'Length Buffer: {len(buffer1[0])}')
        
        # save first recording in first file
        if ((i+ frame_num//2)% frame_num == 0): #first slot
            print("Saving first")
            output = speaker_identification(model=model, name='first', buffer=buffer1, sample_rate=SAMPLE_RATE)
            print(output)
            #save_wav('first',buffer1) #CHANGE
            buffer1=[]
        
        # save first recording in first file
        if (i% frame_num == 0): #second slot
            print("Saving second")
            output = speaker_identification(model=model, name='second', buffer=buffer1, sample_rate=SAMPLE_RATE)
            print(output)
            #save_wav('second',buffer2) #CHANGE to speaker identification
            buffer2=[]

        # feature_vector = extract_features(fileName)