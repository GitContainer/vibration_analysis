# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:42:33 2019

@author: Ravi
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

###############################################################################
# Set the working directory
###############################################################################
os.chdir('D:\\pub\\vibration_analysis')
os.listdir('.')

###############################################################################
# read data
###############################################################################
# file list
folder = 'data'
fnames = os.listdir('data')
full_path = os.path.join(folder, fnames[0])
os.startfile(full_path)

# read the file
cols = 'B:D'
skip_rows = 9
raw_data = pd.read_excel(full_path, usecols = cols, skiprows = skip_rows)
raw_data.head()

# sampling information
tp = 10    # sampling period 10s
ns = 1001  # number of samples
t = np.linspace(0, tp, ns)
ti = t[1] - t[0]

# plotting the waveform
for i in range(3):    
    plt.plot(t, raw_data.iloc[:,i])
    plt.title(raw_data.columns[i])
    plt.xlabel('time (s)')
    plt.show()
    
# turning the data with zero mean
raw_data.mean()
raw_data = raw_data.subtract(raw_data.mean())

# plotting fft
for i in range(3):
    fft = np.fft.fft(raw_data.iloc[:,i])
    f= np.fft.fftfreq(ns, ti)
    plt.plot(f[:ns//2], np.abs(fft)[:ns//2]*1/ns)
    plt.xlabel('Frequency (Hz)')
    title = 'FFT: ' + raw_data.columns[i]
    plt.title(title)
    plt.show()








