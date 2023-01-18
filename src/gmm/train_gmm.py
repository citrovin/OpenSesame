import pickle
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

speaker = "Philipp"
# speaker = "Dalim"
# speaker = "Filippo"


def train_model():

    source   = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/src/gmm/training_set/"
    source += speaker+'/'
    dest = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/src/gmm/trained_models/"

    # train_file = "/Users/Pipo/Documents/University/Business_Intelligence/OpenSesame/Speaker-Identification-Using-Machine-Learning/training_set_addition.txt" 

    files = [f for f in listdir(source) if isfile(join(source, f))]
    features = np.asarray(())
    for train_file in files:
        print(train_file)
        sr,audio = read(source + train_file)
        print(sr)
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        print(features.shape)
    print('FEATURES SHAPE: ', features.shape)
    gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
    gmm.fit(features)
        
    # dumping the trained gaussian model
    picklefile = speaker + ".gmm"
    pickle.dump(gmm,open(dest + picklefile,'wb'))
    print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
    features = np.asarray(())

train_model()