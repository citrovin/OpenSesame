import speech_recognition as sr
from datetime import datetime

# Set the duration of the audio sample (in seconds)
DURATION = 2

# Initialize a recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Function to record audio sample
def record_audio(filename):
    # Start Recording
    with sr.Microphone() as source:
        print("Recording...")
        audio = r.record(source, duration=DURATION)
        print("Recording stopped.")
        
    # Saving the audio
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())


data_path = '../../data/'



while True:

    time = datetime.now().strftime("%d_%m_%H_%M_%S")
    file_path = data_path
    
    choice=int(input("\n 1.Record audio for positive training sample \n 2.Record audio for negative training sample\n 3.Record audio for positive test sample \n 4.Record audio for negative test sample\n"))
    if(choice==1):
        file_path += 'train/positive/pos_'
    elif(choice==2):
        file_path += 'train/negative/neg_'
    elif(choice==3):
        file_path += 'test/positive/pos_'
    elif(choice==4):
        file_path += 'test/negative/neg_'
    if(choice>4):
        exit()


    filename = file_path + time + '.wav'

    # Record audio samples
    record_audio(filename)
    
    