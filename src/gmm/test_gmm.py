import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 


from os import listdir
from os.path import isfile, join

# warnings.filterwarnings("ignore")

def calculate_delta(array):
   
    rows,cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def test_model():

    source   = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/src/gmm/testing_set/"  
    modelpath = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/src/gmm/trained_models/"
     
    files = [f for f in listdir(source) if isfile(join(source, f))]
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Load the Gaussian gender Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
    foo = [fname for fname in gmm_files]
    # print(foo)
    # print(speakers)

    for test_file in files:
        sr,audio = read(source + test_file)
        vector   = extract_features(audio,sr)
            
        log_likelihood = np.zeros(len(models)) 
        print(models)
        print(len(models))
        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            print(scores)
            log_likelihood[i] = scores.sum()
            # print(path)
            print(log_likelihood)
            
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)  
#choice=int(input("\n1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n"))

test_model()
