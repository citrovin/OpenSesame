from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import pyaudio
import wave
import sys
import numpy as np

import warnings
warnings.filterwarnings('ignore')


MODEL = "openai/whisper-base"
#path = '../data/input_filippo_en.wav'
chunk = 2048
recording_seconds = 2

print("################### Loading Model ###################")
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
tokenizer = WhisperTokenizer.from_pretrained(MODEL)


pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)


print("################### Starting Microphone ###################")

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

frames=[]
try:
    while 1:

        frames = []
        for i in range(0, int(16000 / chunk * recording_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        #Stop Recording
        wf = wave.open("output.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))
        wf.close()

        #Transcribing
        text = pipe(inputs="output.wav")

        print(text['text'])
except KeyboardInterrupt as e:
    print(e)


print("################### Terminating Program ###################")
stream.close()