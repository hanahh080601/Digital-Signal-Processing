import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
# chuong trinh lam tron 1 tin hieu de khu nhieu 
# dung bo loc lay trung binh cong 3 diem (3-points moving-averaging filter)
# co PTSP: y[n] = 1/3(x[n]+x[n-1]+x[n-2])

L = 51;                                             # do dai tin hieu
n = np.linspace(0, L-1, L)                          # bien thoi gian roi rac
d = 1.5*np.random.randint(L+1, size=L);             # sinh tin hieu Gausian noise d[n] (1.5 la bien do nhieu)
d = (d - min(d)) / (max(d) - min(d))
s = np.zeros((L))                                   # sinh tin hieu goc s[n] = 2n(0.9)^n
for i in range(len(n)):
    s[i] = 2*i*(0.9**i)
x = s + d;                                          # tin hieu co nhieu x[n]=s[n]+d[n]


plt.subplots(4, 1, figsize=(25, 15))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.93, 
                        wspace=0.05, hspace=0.99)
plt.subplot(4, 1, 1)
plt.xlabel('Chi so thoi gian n')
plt.ylabel('Bien do')
# ve do thi d[n],s[n],x[n]
plt.plot(n, d, 'r-', label='Gause noise d[n]')
plt.plot(n, s, 'k--', label='Origin signal s[n]')
plt.plot(n, x, 'b-.', label='x[n]')
plt.title('Noise d[n] vs. original s[n] vs. noisy signals x[n]');
plt.legend(loc='best')


# cach 1: dich thoi gian, lam tron tin hieu theo CT y[n] = 1/3(x[n-1]+x[n]+x[n+1])
x1 = x.copy()                   # x1[n] = x[n]
x2 = x[:L-1]                    # x2[n] = x[n-1]
z2 = [0]
x2 = np.append(z2, x2)
x3 = x[:L-2]                    # x3[n] = x[n-2]
z3 = np.zeros((2))
x3 = np.append(z3, x3)

plt.subplot(4, 1, 2)
plt.xlabel('Chi so thoi gian n')
plt.ylabel('Bien do')
# ve do thi x1[n],x2[n],x3[n]
plt.plot(n, x1, 'r-', label='x1[n]')
plt.plot(n, x2, 'k-', label='x2[n]')
plt.plot(n, x3, 'b-', label='x3[n]')
plt.title('Time-shifted signals of x[n]');
plt.legend(loc='best')


y1 = 1/3*(x1+x2+x3)     # lay trung binh voi M=1
plt.subplot(4, 1, 3)
plt.xlabel('Chi so thoi gian n')
plt.ylabel('Bien do')
# ve do thi y1[n], s[n]
plt.plot(n, y1, 'r-', label='y1[n]')
plt.plot(n, s, 'b-', label='s[n]')
plt.title('3-points smoothed y1[n] vs. original signal s[n]');
plt.legend(loc='best')


# cach 2: dung ham tinh tong chap conv()
# ghep noi tiep he som 1 don vi va he lay TB cong nhan qua
#h = 1/3 * np.ones(shape=(1,3));        # h[n] = [1/3, 1/3, 1/3]
h = [1/3, 1/3, 1/3]
y2 = np.convolve(x1, h, 'same');         # y2[n] = x1[n] * h[n]


# cach 3: dung ham filter()
#b = 1/3 * np.ones(shape=(1,3));        # b[n] = [1/3, 1/3, 1/3]
b = [1/3, 1/3, 1/3]
a = 1
y3 = lfilter(b, a, x);         
plt.subplot(4, 1, 4)
plt.xlabel('Chi so thoi gian n')
plt.ylabel('Bien do')

# ve do thi y1[n], y2[n], y3[n]
plt.plot(n, y1, 'r.', label='y1[n]')
plt.plot(n, y2, 'b--', label='y2[n]')
plt.plot(n, y3, 'g-.', label='y3[n]')
plt.title('Cach 1 vs Cach 2 vs Cach 3');
plt.legend(loc='best')


plt.show()