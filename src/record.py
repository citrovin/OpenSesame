import speech_recognition as sr

# Set the duration of the audio sample (in seconds)
DURATION = 5

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

# Record audio samples
record_audio("sample1.wav")
record_audio("sample2.wav")
record_audio("sample3.wav")
