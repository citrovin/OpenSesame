# OpenSesame
OpenSesame is a software for speaker identification and speach recognition system. It leverages a Neural Network and a Support Vector Machine to identify if the correct speaker said the correct keyword, such as "open sesame". After detecting the speaker and the key word, the program is to unlock the data, lock, or whatever is connected to the software.

## How does OpenSesame work?
OpenSesame combines two machine learning approaches: 1) NN and 2) SVM, as shown in the following figure.





## Project Structure
- Main
- Data preprocessing for training
- Training the models


## Training Data

Training Samples:
- OS female: 4
- OS negative male: 5
- OS positive male: 8
- Random female: 10
- Random male: 10

Validation Samples:
- OS negative female: 1
- OS negative male: 4
- OS positive male: 4
- Random negative female: 1
- Random negative male: 6

