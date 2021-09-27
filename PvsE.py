from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# read wav file using wavfile.read
Fs, data = wavfile.read('E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\LA001.wav')

data = data - data.min()
data = data / (data.max() - data.min())
data = (data - 0.5) * 2
'''

nb_bits = 0
if data.dtype == 'int16':
    nb_bits = 16  # -> 16-bit wav files
elif data.dtype == 'int32':
    nb_bits = 32  # -> 32-bit wav files
max_nb_bit = float(2 ** (nb_bits - 1))
samples = data / (max_nb_bit + 1)  # samples is a numpy array of floats representing the samples 
'''  
print("Sample rate: ", Fs)
#print("Data: ", data.astype('float32'))
print("X:", data)

# create n[0 1 2 3...]
n = np.linspace(0, data.shape[0] - 1, data.shape[0]).astype('float32')

# plot x[n]
plt.figure(figsize=(15,5))
plt.plot(n, data)


E = np.sum(abs(data)**2)
P = E/data.shape[0]

sd.play(data, Fs)
print("Năng lượng E:", E)
print("Công suất P:", P)
