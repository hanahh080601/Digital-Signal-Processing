from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# read wav file using wavfile.read
Fs, data = wavfile.read('E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\LA001.wav')

print(Fs)
print(data)

# create n[0 1 2 3...]
n = np.linspace(0, data.shape[0] - 1, data.shape[0]).astype(int)

# plot x[n]
plt.figure(figsize=(15,5))
plt.plot(n, data)

E = np.sum(abs(data**2))
P = E/data.shape[0]

print(data.shape[0], P, E)
