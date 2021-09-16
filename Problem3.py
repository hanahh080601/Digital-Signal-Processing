from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from playsound import playsound

Fs, data = wavfile.read('E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\LA001.wav')
playsound('E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\LA001.wav')

print(Fs)
print(data.shape)

n = np.linspace(0, data.shape[0] - 1, data.shape[0]).astype(int)

plt.figure(figsize=(15,5))
plt.stem(n, data, use_line_collection=True)

Fs1 = Fs // 2
Fs2 = Fs * 2

wavfile.write("E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\div2.wav", Fs1, data)
wavfile.write("E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\mul2.wav", Fs2, data)

playsound("E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\div2.wav")
playsound("E:\DUT\Kì 5\Xử lí THS\Tín hiệu mẫu\mul2.wav")
