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
            print("Vector had a different size")
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

        # self.x_train_tensor = open(training_path+'x_train_tensor.npy','w')
        # self.y_train_tensor = open(training_path+'y_train_tensor.npy','w')
        # self.x_test_tensor = open(validation_path+'x_test_tensor.csv', 'w')
        # self.y_test_tensor = open(validation_path+'y_test_tensor.csv', 'w')

    def closeFiles(self):
        self.x_train.close()
        self.y_train.close()
        self.x_test.close()
        self.y_test.close()

        # self.x_train_tensor.close()
        # self.y_train_tensor.close()
        # self.x_test_tensor.close()
        # self.y_test_tensor.close()

if __name__== "__main__" :
    f = Files()
    f.openFiles()

    for dir in dirs:
        files = [dir+f for f in listdir(dir) if isfile(join(dir, f))]
        print(files)
        if dir.find('positive') != -1:
            label = 1
        else:
            label = 0
        
        x, y = createFeatureArray(files, label)
        x_tensor, y_tensor = createFeatureTensor(files, label)
        print(x.shape, y.shape)
        print(x_tensor.shape, y_tensor.shape)

        if dir.find('train') != -1:
            np.savetxt(f.x_train, x, delimiter=',')
            np.savetxt(f.y_train, y, delimiter=',')
            # np.savetxt(f.x_train_tensor, x_tensor.reshape(x_tensor.shape[0], -1), delimiter=',')
            # np.savetxt(f.y_train_tensor, y_tensor.reshape(x_tensor.shape[0], -1), delimiter=',')
            np.save(training_path+'x_train_tensor.npy', x_tensor)
            np.save(training_path+'y_train_tensor.npy', y_tensor)
        else:
            np.savetxt(f.x_test, x, delimiter=',')
            np.savetxt(f.y_test, y, delimiter=',')
            # np.savetxt(f.x_test_tensor, x_tensor.reshape(x_tensor.shape[0], -1), delimiter=',')
            # np.savetxt(f.y_test_tensor, x_tensor.reshape(x_tensor.shape[0], -1), delimiter=',')
            np.save(validation_path+'x_test_tensor.npy', x_tensor)
            np.save(validation_path+'y_test_tensor.npy', y_tensor)
    
    f.closeFiles()

    foo = np.loadtxt(training_path+'x_train.csv', delimiter=',')
    # foo_reshape = foo.reshape(foo.shape[0], foo.shape[1] // 40, 40)
    bar = np.load(training_path+'x_train_tensor.npy', allow_pickle=True)
    print(foo.shape)
    print(bar.shape)

    # for dir in listdir(file_path):
    #     for subdir in listdir(join(file_path, dir)):
    #         print(join(file_path, dir, subdir))
    #         for subsubdir in join(file_path, dir, subdir):
    #             print(join(file_path, dir, subdir, subsubdir))
    #             full_path = [f for f in subdir if isfile(join(file_path, dir, subdir, subsubdir, f))]
    #         # for file in subdir if isfile(join(train_pos, file)):
    #         #     full_path = join(file_path, dir, subdir, file)
    #         print(full_path)
    #         # print(full_path.find('positive') != -1)
    # # for f in listdir()

    train_pos_files = [train_pos+f for f in listdir(train_pos) if isfile(join(train_pos, f))]
    train_neg_files = [train_neg+f for f in listdir(train_neg) if isfile(join(train_neg, f))]
    val_pos_files = [val_pos+f for f in listdir(val_pos) if isfile(join(val_pos, f))]
    val_neg_files = [val_neg+f for f in listdir(val_neg) if isfile(join(val_neg, f))]

    # for file in train_pos_files:
    # pos_x, pos_y = createFeatureArray(train_pos_files, 1)
    # print(pos_x.shape)
    # print(pos_y.shape)
    # pos_x_tensor, pos_y_tensor = createFeatureTensor(train_pos_files, 1)
    # print(np.array(pos_x_tensor).shape)
    # print(pos_y_tensor.shape)

    # neg_x, neg_y = createFeatureArray(train_pos_files, 0)
    # print(neg_x.shape)
    # print(neg_y.shape)
    # neg_x_tensor, neg_y_tensor = createFeatureTensor(train_neg_files, 0)
    # print(np.array(neg_x_tensor).shape)
    # print(neg_y_tensor.shape)

    # # for file in train_pos_files:
    # pos_x, pos_y = createFeatureArray(train_pos_files, 1)
    # print(pos_x.shape)
    # print(pos_y.shape)
    # pos_x_tensor, pos_y_tensor = createFeatureTensor(train_pos_files, 1)
    # print(np.array(pos_x_tensor).shape)
    # print(pos_y_tensor.shape)

    # # for file in train_neg_files:
    # neg_x, neg_y = createFeatureArray(train_pos_files, 0)
    # print(neg_x.shape)
    # print(neg_y.shape)
    # neg_x_tensor, neg_y_tensor = createFeatureTensor(train_neg_files, 0)
    # print(np.array(neg_x_tensor).shape)
    # print(neg_y_tensor.shape)

    # # open and save to csv file x_train
    # # open and save to csv file y_train

    # # for file in val_pos_files:
    # pos_x_val, pos_y_val = createFeatureArray(val_pos_files, 0)
    # print(pos_x_val.shape)
    # print(pos_y_val.shape)
    # pos_x_val_tensor, pos_y_val_tensor = createFeatureTensor(val_neg_files, 0)

    # print(np.array(pos_x_val_tensor).shape)
    # print(pos_y_val_tensor.shape)
    # # for file in val_neg_files:
    # neg_x, neg_y = createFeatureArray(train_pos_files, 0)
    # print(neg_x.shape)
    # print(neg_y.shape)
    # neg_x_tensor, neg_y_tensor = createFeatureTensor(train_neg_files, 0)
    # print(np.array(neg_x_tensor).shape)
    # print(neg_y_tensor.shape)

    

    # TODO shuffle training vectors before saving them?


    