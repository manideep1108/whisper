import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Available Types:  tiny, base, small, medium, large, and .en versions for english only models
model_size = "medium" 
wav_file = "./test.wav"
output_file = "./result.txt"

# Sampling frequency
freq = 44400
 
# Recording duration in seconds
duration = 5

recording = sd.rec(int(duration * freq),
                   samplerate = freq, channels = 2)
 
# Wait for the audio to complete
sd.wait()
 
# using scipy to save the recording in .wav format
# This will convert the NumPy array
# to an audio file with the given sampling frequency
write(wav_file, freq, recording)


model = whisper.load_model(model_size)
result = model.transcribe(wav_file)

with open(output_file, 'w') as f:
    f.write(result["text"])