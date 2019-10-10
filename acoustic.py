#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:35:37 2019

@author: aahunter
"""
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile # get the api
import numpy as np
fs, data = wavfile.read('./data/audio/ZOOM0010.WAV') # load the data
# trim data to 125k-250k
data = data[100000:250000,:]
a = data.T[0] # this is a two channel soundtrack, I get the first track
plt.plot(a,'g') 
plt.show()
#b=[(ele/2**16.) for ele in a] # this is 16-bit track, b is now normalized on [-1,1)
#plt.plot(b,'g') 
c = fft(a) # calculate fourier transform (complex numbers list)
d = int(len(c)/32)  # you only need half of the fft list (real signal symmetry)
k = np.arange(len(data))
T = len(data)/fs  #  fs is the sampling frequency
frqLabel = k/T 
plt.semilogy(frqLabel[:(d-1)],abs(c[:(d-1)]),'r') 
plt.xlabel('frequency Hz')
plt.show()
