from tensorflow import keras

import numpy as np
import pyaudio
import wave
import pickle
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

from utils.preprocess_data import loadData, extract_features

import warnings
warnings.filterwarnings(action='ignore')
counter = 0

# call model
def speaker_identification(
                model, #specify
                name,
                buffer
                ):
    # save recorded files
    path_live_data = save_wav(name,buffer)
    # extract the features and create vectors
    features = extract_features(path_live_data)

    # Feed-forward
    output_nn = model[0].predict(features, verbose=0)
    output_svm = model[1].predict(features)

    score_nn = np.mean(output_nn)
    score_svm = np.mean(output_svm)

    score = np.mean([score_nn, score_svm])

    THRESHOLD_TOTAL = np.mean([THRESHOLD_NN, THRESHOLD_SVM])

    # print(f'SVM: {score_svm} | NN: {score_nn} | Total: {score} | Threshold: {THRESHOLD_TOTAL}')
    global counter
    if counter == 1:
        print("Lock the door again")
        with open('ascii_art/closed_lock.txt', 'r') as f:
            l = f.read()
            print(l)
            f.flush()
            
    if counter > 0:
        counter-=1
        
    if (score_svm>THRESHOLD_SVM and score_nn>THRESHOLD_NN and counter==0):
        counter=3
    # if score>THRESHOLD_TOTAL:
        # print('OpenSesame')
        os.system('clear')
        with open('ascii_art/open_lock.txt', 'r') as f:
            l = f.read()
            print(l)
            f.flush()
         # OVERFLOW WHEN INCLUDING THIS CODE
        # time.sleep(2)
        # os.system('clear')
        # print("Lock the door again")
        # with open('ascii_art/closed_lock.txt', 'r') as f:
        #     l = f.read()
        #     print(l)
        #     f.flush()
            # time.sleep(10)
    else:
        #print(score)
        pass

    

    return 0 #output

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
    THRESHOLD_NN = 0.6
    THRESHOLD_SVM = 0.75
    MODEL_PATH = './feed-forward/models/dense-nn-sr48000-epochs40-v8'
    
    model_nn = keras.models.load_model(MODEL_PATH)  
    
    with open(r"svm/models/svmV2.pickle", "rb") as input_file:
        model_svm = pickle.load(input_file)
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
    with open('ascii_art/closed_lock.txt', 'r') as f:
        l = f.read()
        print(l)
        f.flush()
    i = 0
    while(1):
        i+=1
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer1.append(data)
        buffer2.append(data)       
        buffer3.append(data)
        
        
        if ((i+ (2*frame_num)//3)% frame_num == 0):
            output = speaker_identification(model=[model_nn, model_svm], name='first', buffer=buffer1)
            buffer1=[]

        # save first recording in first file
        if ((i+ (1*frame_num)//3)% frame_num == 0):
            output = speaker_identification(model=[model_nn, model_svm], name='second', buffer=buffer2)
            buffer2=[]
        
        # save second recording in second file
        if (i% frame_num == 0):
            output = speaker_identification(model=[model_nn, model_svm], name='third', buffer=buffer3)
            buffer3=[]
