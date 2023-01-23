from tensorflow import keras

import numpy as np
import pyaudio
import wave

from utils.preprocess_data import loadData, extract_features

import warnings
warnings.filterwarnings(action='ignore')

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
    features = extract_features(path_live_data) # return a np.array with the features
    #print(features.shape)
    #features = np.swapaxes(features, 0, 1) # swap axis of features
    #print(features.shape)
    
    # predict the speaker or not
    # RNN
    #output = model.predict(np.expand_dims(features, axis=0))

    # Feed-forward
    output = model.predict(features)
    
    # Choose if it is correct speaker
    
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
    
    MODEL_PATH = './feed-forward/models/dense-nn-sr48000-epochs100-v3'
    
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
        #print(f'Length Buffer: {len(buffer1[0])}')
        
        # save first recording in first file
        if ((i+ frame_num//2)% frame_num == 0): #first slot
            #print("Saving first")
            output = speaker_identification(model=model, name='first', buffer=buffer1, sample_rate=SAMPLE_RATE)
            print(np.mean(output))
            #save_wav('first',buffer1) #CHANGE
            buffer1=[]
        
        # save first recording in first file
        if (i% frame_num == 0): #second slot
            #print("Saving second")
            output = speaker_identification(model=model, name='second', buffer=buffer2, sample_rate=SAMPLE_RATE)
            print(np.mean(output))
            #save_wav('second',buffer2) #CHANGE to speaker identification
            buffer2=[]

        # feature_vector = extract_features(fileName)