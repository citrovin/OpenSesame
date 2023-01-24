# OpenSesame
OpenSesame is a software for speaker identification and speach recognition system. It leverages a Neural Network and a Support Vector Machine to identify if the correct speaker said the correct keyword, such as "open sesame". After detecting the speaker and the key word, the program is to unlock the data, lock, or whatever is connected to the software.

## How does OpenSesame work?
OpenSesame is implemented with the logic shown in Fig. 1. 

![](./images/logic.png "Figure 1: Logic of OpenSesame")

It consists of 4 parts: 1) live recording, 2) NN, 3) SVM, and 4) the decision block. In the following, there is a short paragraph about the technology behind OpenSesame.

### Live Recording
OpenSesame records live data to identify if the keyword has been said by the correct user in real time. The recordings are saves in three wave files, that overlap to catch the case if the keyword is split onto two different recordings. The prediction is then run over the three recordings.

### Prediction of Speaker and Keyword
OpenSesame combines two machine learning approaches: 1) NN and 2) SVM, as shown in the following Firgure 1. The program does a live recording and predicts the both scores of the NN and the SVM if it is the correct speaker and the correct keyword. Only if both scores exceed a certain threshold, the opening of the program is triggered, as shown in Fig. 2.

![](./images/OpenSesame.gif "Figure 2: Unlock Screen")

#### Neural Network Archtecture
The Neural Network used for OpenSesame is a Feed-Forward neural network, that implents 6 Dense layers. The first layer expands the feature vector from 40 to 256 dimensions. Every following layer decreases the dimensionality by a power of 2, namely 128, 64, 32, 16, 1.

### Support Vector Machine


## Training Data
For training we collected 136 recordings, which is split into 50% positive and 50% negative samples. Each recording is split into 197 vectors, which are fed to the model during training. For testing our model, we use the exact same strateg as for training.

|   | Recordings for Training | Recordings for Testing |
| ------------- | ------------- | ------------- |
| Positive  | 68  | 13 |
| Negative  | 68  | 13 |
| Total  | 136  | 26 |



## Project Structure
- Main
- Data preprocessing for training
- Training the models


### Team

The team behind this project consists of three graudate engineering students, currently enrolled in the EIT Autonomous Systems program at Polytech Nice-Sophia.


1) Filippo Zeggio
2) Philipp Ahrendt
3) Dalim Wahby
