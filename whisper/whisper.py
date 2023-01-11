from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import pyaudio
import wave
import sys
import numpy as np


MODEL = "openai/whisper-base"
#path = '../data/input_filippo_en.wav'


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





#output = pipe(inputs=path)
#print(output)
frames=[]
try:
    while 1:
        data = stream.read(4096)

        data_np = np.frombuffer(data, np.float32)
        # data = 'RIFF$t\r\x00WAVEfmt' + str(data)
        #print(type(data))
        #print("./output.wav")
        # print(frames)
        text = pipe(inputs=data_np)
        print(text)
except KeyboardInterrupt as e:
    print(e)


print("################### Terminating Program ###################")
stream.close()