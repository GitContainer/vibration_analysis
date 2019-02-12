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
ftype = 'csv'
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
vib = csv_vibs['30645__raw data acceleration']

n, _ = vib.shape
d = vib.iloc[1,0] - vib.iloc[0,0]

# visualize the data before plotting
n_disc = 100
for i in range(1,4):
    plt.plot(vib.iloc[n_disc:,0], vib.iloc[n_disc:,i])
    plt.show()
    
###############################################################################
# get fft
###############################################################################
def get_fft(i, n_disc, raw_data):
    t = raw_data.iloc[n_disc:,0]
    s = raw_data.iloc[n_disc:,i]
    
    n = len(t)
    d = t.iloc[1] - t.iloc[0]
    
    
    fft = np.fft.fft(s)
    f= np.fft.fftfreq(n, d)
    
    return f, fft

def plot_frequency_spectrum(n_disc, raw_data, name):
    fig, ax = plt.subplots(3, 1)
    
    for i in range(1,4):
        f, fft = get_fft(i, n_disc, raw_data)
        n = len(f)
        ax[i-1].plot(f[:n//2], np.abs(fft)[:n//2]*1/n)
        ax[i-1].set_ylabel('Acceleration')
        
        if i == 1:
            ax[i-1].set_ylim(0, 0.029)
        
        if i == 3:
            ax[i-1].set_ylim(0, 0.06)
            
    ax[2].set_xlabel('Frequency (Hz)')
    ax[0].set_title('FFT: ' + name)
    
    plt.subplots_adjust(bottom=0.0, right=0.8, top=0.9)
    return ax
        
       
for key in csv_vibs:
    print(key)
    raw_data = csv_vibs[key]
    ax = plot_frequency_spectrum(n_disc, raw_data, key)
    
n_disc = 300
key = '27741_raw data acceleration'                
raw_data = csv_vibs[key]
ax = plot_frequency_spectrum(n_disc, raw_data, key)    



###############################################################################
# getting velocity spectrum
###############################################################################    
    




    
    
    
    
    
    
    
    
    
    
    