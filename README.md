# **OpenSesame: The speaker recognition system, that keeps the integrity of your data!**


OpenSesame is a software for speaker identification and speach recognition system. It leverages machine lerning, namely a Neural Network and a Support Vector Machine to identify if the correct speaker said the correct keyword, such as "open sesame". After detecting the speaker and the key word, the program is to unlock the data, lock, or whatever is connected to the software.

---

### **How does OpenSesame work?**
OpenSesame consists of 4 parts: 1) live recording, 2) neural network, 3) support vector machine, and 4) the decision block, as shown in Fig. 1. In the following, we will shortly introduce the main components of the program.

![](./images/logic.png "Figure 1: Logic of OpenSesame")
Figure 1: Logic of OpenSesame

1) Live Recording: OpenSesame records live data to identify if the keyword has been said by the correct user in real time. The recordings are saves in three wave files, that overlap to catch the case if the keyword is split onto two different recordings. The prediction is then run over the three recordings.

2) Neural Network (NN): The neural network computes a prediction value of between 0 and 1. If it surpasses the threshold of 0.6, the system recognises that the correct speaker said the correct keyword.

3) Support Vector Machine (SVM): The SVM computes a prediction value of between 0 and 1. If it surpasses the threshold of 0.75, the system recognises that the correct speaker said the correct keyword.

4) Decision: If both thresholds of the NN and the SVM are supassed, then and only then the system unlocks. Using two different models, gives us a kind of fail-safe, for the case that the NN or the SVM somehow predicts a high value, eventhoug it should not have. The unlock screen, as shown in Fig. 2.

![](./images/OpenSesame.gif "Figure 2: Unlock Screen")
Figure 2: Unlock Screen

---

### **Machine Learning Models**

#### **Neural Network Archtecture**
The Neural Network used for OpenSesame is a Feed-Forward neural network, that implents 6 Dense layers. The first layer expands the feature vector from 40 to 256 dimensions. Every following layer decreases the dimensionality by a power of 2, namely 128, 64, 32, 16, 1. All layers use a Relu activation function, except the last layer. It uses a Sigmoid activation, which gives us a value that represents a probability (bewteen 0 and 1) if the correct speaker said the correct keyword. The model is trained on 40 epochs. The training results can be seen in the following parts:

![](./src/feed-forward/plots/plots_dense-nn-sr48000-epochs40-v8.png "Figure 3: Accuracy and Loss of the Training process")
Figure 3: Accuracy and Loss of the Training process


Since we unpack every recording into 197 individual vectors, we evaluate the trained model twice. Firstly, how well did the model generalise on all samples and how well did the model generalise on the recodings?


Figure 4: Confusion Matrix of Prediction on all samples | Figure 5: Confusion Matrix of Prediction on all recordings
:-------------------------:|:-------------------------:
![](./src/feed-forward/plots/heatmap_samples_dense-nn-sr48000-epochs40-v9.png "Figure 4: Confusion Matrix of NN Prediction on all samples")  |  ![](./src/feed-forward/plots/heatmap_recordings_dense-nn-sr48000-epochs40-v9.png "Figure 5: Confusion Matrix of NN Prediction on all recordings")




Performance Metrics on individual samples:


```
Classificaiton Report over all sample vectors:
              precision    recall  f1-score   support

         0.0       0.68      0.78      0.73      2919
         1.0       0.81      0.72      0.76      3779

    accuracy                           0.75      6698
   macro avg       0.75      0.75      0.74      6698
weighted avg       0.75      0.75      0.75      6698
```


Performance Metrics on recordings:

```
Classificaiton Report over all recordings:
              precision    recall  f1-score   support

         0.0       0.88      1.00      0.94        15
         1.0       1.00      0.89      0.94        19

    accuracy                           0.94        34
   macro avg       0.94      0.95      0.94        34
weighted avg       0.95      0.94      0.94        34
```


As we can see our model performs well on predicting, if an entire recording contains the correct keyword, by the correct user. Thereofre, it is not as important to look at the results of the classification of every sample.



#### **Support Vector Machine**
The SVM model implements the default ```SVC()``` model provided by the sklearn library, which uses a radial basis function as the kernel. After training we get the following results:


Figure 5: Confusion Matrix of Prediction on all samples | Figure 6: Confusion Matrix of Prediction on all recordings
:-------------------------:|:-------------------------:
![](./src/svm/plots/heatmap_samples_svmV2.png "Figure 6: Confusion Matrix of SVM Prediction on all samples")  |  ![](./src/svm/plots/heatmap_svm_recordings_svmV2.png "Figure 6: Confusion Matrix of SVM Prediction on all recordings")




Performance metrics on all sample:
```
Classificaiton Report over all sample vectors:
              precision    recall  f1-score   support

         0.0       0.69      0.76      0.72      3042
         1.0       0.78      0.71      0.75      3656

    accuracy                           0.73      6698
   macro avg       0.73      0.74      0.73      6698
weighted avg       0.74      0.73      0.74      6698
```

Performance metrics on recordings:
```
Classification Report over all recordings:
              precision    recall  f1-score   support

         0.0       0.76      1.00      0.87        13
         1.0       1.00      0.81      0.89        21

    accuracy                           0.88        34
   macro avg       0.88      0.90      0.88        34
weighted avg       0.91      0.88      0.88        34
```

The SVM does not classify as well on the test data, compared to the the NN. For our use-case it does not have to perform as well, as the neural network. However, it is still necessary to integerate the models for safety reasons. If we were to only rely on the NN, the probability that unathorised access is granted is higher, in case of a false positive. In contrast, if we use both models, they **both** have to classify the speaker as the correct one and identify the keyword. This feature gives us additional protection against possible intruders.


#### **Training Data**
For training we collected 156 recordings, which is split into 50% positive and 50% negative samples. Each recording is split into 197 vectors, which are fed to the model during training. This results in 30,732 training samples per epoch. For testing our model, we use the exact same strategy as for training.


|   | Recordings for Training | Recordings for Testing |
| ------------- | ------------- | ------------- |
| Positive  | 78  | 17 |
| Negative  | 78  | 17 |
| Total  | 156  | 34 |

---

### **Project Structure**

```
├── README.md
├── data
│   ├── live
│   ├── test
│   └── train
├── images
├── requirements.txt
└── src
    ├── ascii_art
    │   ├── closed_lock.txt
    │   └── open_lock.txt
    ├── feed-forward
    │   ├── feed-forward_train.py
    │   ├── models
    │   ├── old
    │   └── plots
    ├── gmm
    │   ├── test_gmm.py
    │   ├── testing_set
    │   ├── train_gmm.py
    │   ├── trained_models
    │   └── training_set
    ├── main.py
    ├── rnn
    │   ├── models
    │   ├── plots
    │   ├── rnn-validate.py
    │   └── rnn_train.py
    ├── svm
    │   ├── models
    │   ├── old
    │   ├── plots
    │   └── svm_train.py
    └── utils
        ├── __init__.py
        ├── im2a.py
        ├── preprocess_data.py
        └── record.py
```

All code is stored in the ```./src``` directory. It contains the ```main.py``` file, that contains the actual program, implementing the logic and the used models. Additionally, there are further models (i.e. ```./src/rnn``` ), that were used for training, however were not seen as feasible due to computation costs or other factors that have to be taken into account.

Every directory named after a model, i.e. ```./src/feed-forward``` contain the training file of the models and the trained models, that are used in ```main.py```.

The ```./src/utils``` directory contains all the files needed to record and preprocess our training and testing data.

Lastly, the ```./requirements.txt``` files specify which libraries we used, if you want to use this repository.

---

### **Team**

This project is part of our graduate program, as part of the Business Intelligence Lecture. The team behind it consists of three graudate engineering students, currently enrolled in the EIT Autonomous Systems program at Polytech Nice-Sophia.

- Filippo Zeggio: https://github.com/curcuman
- Philipp Ahrendt: https://github.com/phiahr
- Dalim Wahby: https://github.com/citrovin