from deepspeech import Model
from scipy.io import wavfile
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='5,6'

audio_path = '.../aud.wav'
fs, data = wavfile.read(audio_path)
model_path = ".../.tensorflow/models/deepspeech-0_1_0-b90017e8.pb" 
ars = Model(model_path) 
translate_txt = ars.stt(data)
print(translate_txt)
