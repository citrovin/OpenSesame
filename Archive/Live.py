import pyaudio
import wave
import numpy as np
import time

# Constants. You can decrease the chunk if you want a faster loop (faster sample rate)
chunk = 2048
recording_seconds = 2

def save(name,buffer):
        with wave.open(name+'.wav', 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"".join(buffer))

print("Opening stream..")
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk)
stream.start_stream()


buffer1=[]
buffer2=[]
frame_num =int( ((16000 / chunk) * recording_seconds) )
print(frame_num)

i = 0
print("Let's go! Good job Filippo!")
for h in range(0,100):
    i+=1
    data = stream.read(chunk)
    buffer1.append(data)
    buffer2.append(data)
    print(i)
  
    if ((i+ frame_num//2)% frame_num == 0): #first slot
        print("Saving first")
        save('first',buffer1)
        buffer1=[]

    if (i% frame_num == 0): #second slot
        print("Saving second")
        save('second',buffer2)
        buffer2=[]


