# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter, freqz, freqs
import matplotlib.pyplot as plt
from pylab import *
import scipy.signal as signal

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Stack Exchange/data1.csv", sep=';');

ts = 0.02
fs = 1/ts 
data = df.iloc[:,2]
t = np.arange(0,len(data))*ts

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(25,3))

ax1.plot(t,data, 'r', linestyle = '-')
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time[s]")
ax1.grid()

ax2.plot(t,data, 'r', linestyle = '-')
ax2.set_ylabel("Amplitude")
ax2.set_xlabel("Time[s]")
ax2.set_xlim((0,10))
ax2.grid()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
################################################################################
# Filter requirements.
order1 = 4
cutoff = 0.1      # desired cutoff frequency of the filter, Hz
################################################################################
# Get the filter coefficients so we can check its frequency response.
b1, a1 = butter_lowpass(cutoff, fs, order1)

"""## IIR"""

N = order1
L = len(data)
x = np.zeros(L,dtype=np.float64)
u = np.zeros(L,dtype=np.float64)
y = np.zeros(L,dtype=np.float64)
a = a1
b = b1

for n in range(L):

  for i in range(N,1,-1):
    x[n-i-1] = x[n-i]
  x[n] = data[n]

  y[n] = 0
  for i in range(N):
    y[n] = y[n] + b[i]*x[n-i]

  for i in range(1,N):
    y[n] = y[n] + a[i]*y[n-i]
