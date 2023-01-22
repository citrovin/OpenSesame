import numpy as np
from scipy.io.wavfile import read
from os import listdir
from os.path import isfile, join
import python_speech_features as mfcc
from sklearn import preprocessing


##### PATH DEFINITIONS #####
file_path = "../data/"

training_path = file_path + "train/"
validation_path = file_path + "validation/"

extension = '.wav'

# labels = ['positive', 'negative']

pos = 'positive/'
neg = 'negative/'


##### VAR DEFINITIONS #####
train_pos = training_path+pos
train_neg = training_path+neg
val_pos = validation_path+pos
val_neg = validation_path+neg

dirs = [train_pos, train_neg, val_pos, val_neg]

# x_train = None
# y_train = None
# x_test  = None
# y_test  = None

# x_train_tensor = None
# y_train_tensor = None
# x_test_tensor  = None
# y_test_tensor  = None

# for label in labels:

def calculate_delta(array):
   
    rows,cols = array.shape
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



def exportFeatures():
    pass

#TODO create feature array to import with argument depending on what kind of feature data you want
# feature_extraction_method = ['simple', 'with_delta', 'different_sample_rate', 'as_row_vectors', 'as_matrices']

def importFeatures():
    pass

#TODO
def getTrainingData():
    features = []
    labels = []
    return (features, labels)

def getValidationData():
    pass

def createFeatureArray(files, label):
    
    features = np.asarray(())
    # features = []

    for file in files:
        sr,audio = read(file)
        vector = extract_features(audio,sr)
        # vector = extract_features(audio,sample_rate)
        # print(vector.shape)
        # vector = extract_mfcc(file_path+file+".wav")
        # print('file:', file, features.shape)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))            
        # features.append(vector)
    if label == 1:
        labels = np.ones(features.shape[0])
    else:
        labels = np.zeros(features.shape[0])
    
    return (features,labels)

def createFeatureTensor(files, label):
    
    features = []
    vector_shape = None

    for file in files:
        sr,audio = read(file)
        vector = extract_features(audio,sr)
        if len(features) == 0:
            vector_shape = vector.shape
        if vector_shape != vector.shape:
            print("Vector had a different size, file: ", file)
            continue
        features.append(vector)
        
    
    features = np.array(features)
    
    if label == 1:
        labels = np.ones(features.shape[0])
    else:
        labels = np.zeros(features.shape[0])

    labels = np.expand_dims(np.expand_dims(labels, axis=1), axis=1)

    return (features, labels)
    

def shuffleSampleVectors():
    pass


class Files:
    def __init__(self):
        pass

    def openFiles(self):
        self.x_train = open(training_path+'x_train.csv','w')
        self.y_train = open(training_path+'y_train.csv','w')
        self.x_test = open(validation_path+'x_test.csv', 'w')
        self.y_test = open(validation_path+'y_test.csv', 'w')

        self.x_train_tensor = open(training_path+'x_train_tensor.npy','wb')
        self.y_train_tensor = open(training_path+'y_train_tensor.npy','wb')
        self.x_test_tensor = open(validation_path+'x_test_tensor.csv', 'wb')
        self.y_test_tensor = open(validation_path+'y_test_tensor.csv', 'wb')

    def closeFiles(self):
        self.x_train.close()
        self.y_train.close()
        self.x_test.close()
        self.y_test.close()

        # self.x_train_tensor.close()
        # self.y_train_tensor.close()
        # self.x_test_tensor.close()
        # self.y_test_tensor.close()

class Arrays:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_train_tensor = None
        self.y_train_tensor = None

        self.x_test = None
        self.y_test = None
        self.x_test_tensor = None
        self.y_test_tensor = None

def loadData(asTensor=True):

    if asTensor:
        x_train = np.load(training_path+'x_train_tensor.npy', allow_pickle=True)
        y_train = np.load(training_path+'y_train_tensor.npy', allow_pickle=True)
        x_test = np.load(validation_path+'x_test_tensor.npy', allow_pickle=True)
        y_test = np.load(validation_path+'y_test_tensor.npy', allow_pickle=True)
    else:
        x_train = np.loadtxt(training_path+'x_train.csv', delimiter=',')
        y_train = np.loadtxt(training_path+'y_train.csv', delimiter=',')
        x_test = np.loadtxt(validation_path+'x_test.csv', delimiter=',')
        y_test = np.loadtxt(validation_path+'y_test.csv', delimiter=',')

    return (x_train, y_train, x_test, y_test)

if __name__== "__main__" :
    f = Files()
    f.openFiles()
    arr = Arrays()

    for dir in dirs:
        files = [dir+f for f in listdir(dir) if isfile(join(dir, f))]
        # print(len(files))
        # print(dir)
        if dir.find('positive') != -1:
            label = 1
        else:
            label = 0
        
        x, y = createFeatureArray(files, label)
        x_tensor, y_tensor = createFeatureTensor(files, label)
        # if len(x) == 0 and len(y) == 0:
        #     x, y = createFeatureArray(files, label)
        #     x_tensor, y_tensor = createFeatureTensor(files, label)
        # else:
        #     print("Append x")
        #     tmp_x, tmp_y = createFeatureArray(files, label)
        #     tmp_x_tensor, tmp_y_tensor = createFeatureTensor(files, label)
        #     np.append(x,tmp_x)
        #     np.append(y,tmp_y)
        #     np.append(x_tensor,tmp_x_tensor)
        #     np.append(y_tensor,tmp_y_tensor)
        # print(x.shape, y.shape)
        # print(x_tensor.shape, y_tensor.shape)

        if dir.find('train') != -1:
            if isinstance(arr.x_train, type(None)):
                print('if')
                arr.x_train = x
                arr.y_train = y
                arr.x_train_tensor = x_tensor
                arr.y_train_tensor = y_tensor
            else:
                print('else')
                print(arr.x_train_tensor.shape, x_tensor.shape)
                arr.x_train = np.vstack([arr.x_train, x])
                arr.y_train = np.append(arr.y_train, y)
                arr.x_train_tensor = np.vstack([arr.x_train_tensor, x_tensor])
                arr.y_train_tensor = np.vstack([arr.y_train_tensor, y_tensor])
        else:
            if isinstance(arr.x_test, type(None)):
                arr.x_test = x
                arr.y_test = y
                arr.x_test_tensor = x_tensor
                arr.y_test_tensor = y_tensor
            else:
                arr.x_test = np.vstack([arr.x_test, x])
                arr.y_test = np.append(arr.y_test, y)
                arr.x_test_tensor = np.vstack([arr.x_test_tensor, x_tensor])
                arr.y_test_tensor = np.vstack([arr.y_test_tensor, y_tensor])

    np.savetxt(f.x_train, arr.x_train, delimiter=',')
    np.savetxt(f.y_train, arr.y_train, delimiter=',')
    np.save(f.x_train_tensor, arr.x_train_tensor)
    np.save(f.y_train_tensor, arr.y_train_tensor)

    np.savetxt(f.x_test, arr.x_test, delimiter=',')
    np.savetxt(f.y_test, arr.y_test, delimiter=',')
    np.save(validation_path+'x_test_tensor.npy', arr.x_test_tensor)
    np.save(validation_path+'y_test_tensor.npy', arr.y_test_tensor)
    
    f.closeFiles()

    x_train, y_train, x_test, y_test = loadData()
    

    # TODO shuffle training vectors before saving them?


    