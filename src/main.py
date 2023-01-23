from tensorflow import keras

import numpy as np
import pyaudio
import wave

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

from utils.preprocess_data import loadData, extract_features

import warnings
warnings.filterwarnings(action='ignore')

# call model
def speaker_identification(
                model, #specify
                name,
                buffer
                ):
    # save recorded files
    path_live_data = save_wav(name,buffer)
    
    # extract the features and create vectors
    features = extract_features(path_live_data) # return a np.array with the features
    #print(features.shape)
    #features = np.swapaxes(features, 0, 1) # swap axis of features
    #print(features.shape)
    
    # predict the speaker or not
    # RNN
    #output = model.predict(np.expand_dims(features, axis=0))

    # Feed-forward
    output = model.predict(features, verbose = 0)
    
    # Choose if it is correct speaker
    score = np.mean(output)
    if score>THRESHOLD:
        # print('OpenSesame')
        os.system('clear')
        with open('utils/open_lock.txt', 'r') as f:
            l = f.read()
            print(l)

            # OVERFLOW WHEN INCLUDING THIS CODE
            # time.sleep(3)
            # os.system('clear')
            # with open('utils/closed_lock.txt', 'r') as f:
            #     l = f.read()
            #     print(l)
    else:
        #print(score)
        pass
    
    return output

# save the recorded data in 2 files, asynchronously and call the model on the saved file
def save_wav(name,buffer):
    path_live_data = '../data/live/'+name+'.wav'
    with wave.open(path_live_data, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000) #16000
        f.writeframes(b"".join(buffer))
            
    return path_live_data


if __name__== "__main__" :

    
    # Constants. You can decrease the chunk if you want a faster loop (faster sample rate)
    CHUNK = 2048 # in buffer always 2 times the number of chunk is saved
    RECORDING_SECONDS = 2
    SAMPLE_RATE = 48000
    THRESHOLD = 0.75
    
    MODEL_PATH = './feed-forward/models/dense-nn-sr48000-epochs37-v5-positives'
    
    model = keras.models.load_model(MODEL_PATH)
    print('Model loaded')
    
    print("Opening stream..")
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=2*CHUNK )
    stream.start_stream()


    buffer1=[]
    buffer2=[]
    buffer3=[]
    frame_num = int( ((SAMPLE_RATE / CHUNK) * RECORDING_SECONDS))

    os.system('clear')
    with open('utils/closed_lock.txt', 'r') as f:
        l = f.read()
        print(l)

    i = 0
    while(1):
        i+=1
        data = stream.read(CHUNK)
        buffer1.append(data)
        buffer2.append(data)       
        buffer3.append(data)
        
        
        if ((i+ (2*frame_num)//3)% frame_num == 0):
            output = speaker_identification(model=model, name='first', buffer=buffer1)
            buffer1=[]

        # save first recording in first file
        if ((i+ (1*frame_num)//3)% frame_num == 0):
            output = speaker_identification(model=model, name='second', buffer=buffer2)
            buffer2=[]
        
        # save second recording in second file
        if (i% frame_num == 0):
            output = speaker_identification(model=model, name='third', buffer=buffer3)
            buffer3=[]