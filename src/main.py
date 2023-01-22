from tensorflow import keras

import librosa
import numpy as np
import pyaudio
import wave

import time
import os

def save_wav(name,buffer):
        with wave.open(name+'.wav', 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"".join(buffer))


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
    
    # Constants. You can decrease the chunk if you want a faster loop (faster sample rate)
    chunk = 2048
    recording_seconds = 2
    

    print("Opening stream..")
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk)
    stream.start_stream()


    buffer1=[]
    buffer2=[]
    frame_num =int( ((16000 / chunk) * recording_seconds) )
    print(frame_num)

    i = 0
    print("Let's go! Good job Filippo!")
    while(1):
        i+=1
        data = stream.read(chunk)
        buffer1.append(data)
        buffer2.append(data)
        print(i)
      
        if ((i+ frame_num//2)% frame_num == 0): #first slot
            print("Saving first")
            save_wav('first',buffer1)
            buffer1=[]

        if (i% frame_num == 0): #second slot
            print("Saving second")
            save_wav('second',buffer2)
            buffer2=[]
