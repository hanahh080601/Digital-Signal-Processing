from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
import sounddevice as sd
import time

A = 10
F0 = 1000
phi = 0

Fs1 = 3*F0
print(Fs1)
Ts1 = 1/Fs1
n1 = np.linspace(0,4)
print(n1.shape)
x1 = A*np.cos(2*math.pi*F0*n1*Ts1 + phi).astype(float)
print(x1)

Fs2 = 1.5*F0
print(Fs2)
Ts2 = 1/Fs2
n2 = np.linspace(0,4)
x2 = A*np.cos(2*math.pi*F0*n2*Ts2 + phi).astype(float)

plt.subplot(2,1,1)
plt.plot(x1)

plt.subplot(2,1,2)
plt.plot(x2)


sd.play(x1, Fs1)
time.sleep(1)
sd.play(x2, Fs2)
time.sleep(1)
sd.play(x1, F0)
time.sleep(1)
sd.play(x2, F0)




