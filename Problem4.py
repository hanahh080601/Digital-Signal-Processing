from scipy.io import wavfile
import numpy as np
from math import *
import matplotlib.pyplot as plt

A = 1
F0 = 500
phi = 0

Fs1 = 3*F0
Ts1 = 1/Fs1
n1 = np.linspace(0,Fs1)
x1 = A*np.cos(2*pi*F0*n1*Ts1 + phi)

Fs2 = 1.5*F0
Ts2 = 1/Fs2
n2 = np.linspace(0,Fs2)
x2 = A*np.cos(2*pi*F0*n2*Ts2 + phi)

plt.subplot(2,1,1)
plt.plot(x1)

plt.subplot(2,1,2)
plt.plot(x2)





