from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

Fs, data = wavfile.read('E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\LA001.wav')

print(Fs)
print(data)

n = np.linspace(0, data.shape[0] - 1, data.shape[0]).astype(int)

plt.figure(figsize=(15,5))
plt.plot(n, data)

Fs1 = Fs // 2
Fs2 = Fs * 2

sd.play(data, Fs)
sd.wait()
sd.play(data, Fs1)
sd.wait()
sd.play(data, Fs2)


