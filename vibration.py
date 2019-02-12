# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:05:47 2019

@author: Ravi Tiwari
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import PyPDF2


###############################################################################
# Set the working directory
###############################################################################
os.chdir('D:\\pub')
os.listdir('.')

###############################################################################
# get the list of files in the folder
###############################################################################
def initialize_empty_file_list(folders, file_types):
    file_list = {}
    
    for folder in folders:
        file_list[folder] = {}
        for ftype in file_types:
            file_list[folder][ftype] = []
    return file_list

def add_file_names_to_the_list(prefix, folders, file_list):
    for folder in folders:
        full_path = os.path.join(prefix, folder)
        files = os.listdir(full_path)
        for file in files:
            if '.csv' in file:
                file_list[folder]['csv'].append(file)
            if '.pdf' in file:
                file_list[folder]['pdf'].append(file)
            if '.xlsx' in file:
                file_list[folder]['xl'].append(file)
    return file_list

def get_file_list():    
    prefix = 'D:\\pub\\Vibration data\\Vibration data'
    folders = ['LL202', 'LL204', 'LL206']
    file_types = ['csv', 'pdf', 'xl']
    empty_file_list = initialize_empty_file_list(folders, file_types)
    file_list = add_file_names_to_the_list(prefix, folders, empty_file_list)
    return file_list
          
file_list = get_file_list()


###############################################################################
# Question to answer: does every csv/xl/pdf file has the same structure
# Here I just open all the file and check visually
###############################################################################
def open_all_files_of_given_type(prefix, file_list, ftype):    
    for folder in file_list:
        for filetype in file_list[folder]:
            if filetype == ftype:    # change here to open different type of file (xl)
                filenames = file_list[folder][filetype]
                for filename in filenames:
                    print(filename)
                    print(filename.split('.')[0])
                    full_name = os.path.join(prefix, folder, filename)
                    os.startfile(full_name)
    return


prefix = 'D:\\pub\\Vibration data\\Vibration data'
ftype = 'pdf'
open_all_files_of_given_type(prefix, file_list, ftype)
        
###############################################################################
# save data for csv and xl files in a dictionary
###############################################################################

def store_all_data_in_dict(file_list, filetype = 'csv'):
    vibs = {}
    for folder in file_list:
        for ftype in file_list[folder]:
            if ftype == filetype:    
                filenames = file_list[folder][filetype]
                for filename in filenames:
                    fileid = filename.split('.')[0]
                    full_name = os.path.join(prefix, folder, filename)
                    if filetype == 'csv':
                        vib = pd.read_csv(full_name, skiprows = 2)
                        vibs[fileid] = vib
                    if filetype == 'xl':
                        vib = pd.read_excel(full_name, skiprows = 0)
                        vibs[fileid] = vib
    return vibs
                        
csv_vibs = store_all_data_in_dict(file_list, filetype = 'csv')
xl_vibs = store_all_data_in_dict(file_list, filetype = 'xl')
                    
###############################################################################
# plotting the saved data
###############################################################################

def plot_vib_data(name, vib):
    f, ax = plt.subplots(3, 1, sharex=True)
    for i in range(1,4):
        ax[i-1].plot(vib.iloc[:,0], vib.iloc[:,i])   
        ax[i-1].set_ylabel(vib.columns[i])
    ax[i-1].set_xlabel(vib.columns[0])
    ax[0].set_title(name)
    return ax

def plot_all(vibs):
    for name in vibs:
        vib = vibs[name]
        ax = plot_vib_data(name, vib)
        plt.show()
    return

plot_all(csv_vibs)
plot_all(xl_vibs)


###############################################################################
# Doing fft and plotting the result
###############################################################################    
    



###############################################################################
# Read data (csv and xl will work)
###############################################################################



###############################################################################
# 3. PDF
# As of now, pdf is not working. Try Later
###############################################################################
filetype = 'pdf'
filename = file_list[folder][filetype][0]
full_name = os.path.join(prefix, folder, filename)

os.startfile(full_name)

#pdf_file = open(full_name, 'a')
#read_pdf = PyPDF2.PdfFileReader(pdf_file)

###############################################################################




###############################################################################
# read data shared by palani
###############################################################################
folder = 'D:\\pub\\old data\\Vibration data\\Vibration data\\LL202'
file = '30461__raw data acceleration.csv'
#file = '30645__raw data acceleration.csv'
full_name = os.path.join(folder, file)

vib = pd.read_csv(full_name, skiprows = 2)
vib.head()

for i in range(1,4):
    plt.plot(vib.iloc[100:,0]*100, vib.iloc[100:,i])
    plt.show()    

    

# fft of the data
plt.plot(vib.iloc[100:,1])   
N = vib.iloc[100:,1].shape[0]
T = vib.iloc[101,0] - vib.iloc[100,0]

f = np.linspace(0, 1/T, N)

fft = np.fft.fft(vib.iloc[100:,1])


plt.plot(f[:N//2], np.abs(fft)[:N//2]*1/N)
   
###############################################################################
# Exploring FFT
###############################################################################    
t = np.linspace(0, 0.5, 500)
s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.plot(t, s)
plt.show()    
    
fft = np.fft.fft(s) 

fft = np.fft.fft(s)
T = t[1] - t[0]  # sampling interval 
N = s.size

# 1/T = frequency
f = np.linspace(0, 1 / T, N)

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
plt.show()  


############################################################################### 
# another example
###############################################################################
import matplotlib.pyplot as plt
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real) # freq, sp.imag)
plt.show()    

plt.plot(freq, sp.imag) # freq, sp.imag)
plt.show()    
        
    
    
    
    
    
    
    
    
    
    
    