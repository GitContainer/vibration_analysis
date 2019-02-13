# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:05:47 2019

@author: Ravi Tiwari
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
# Visualize the data before plotting
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
        
# All FFT in one go       
for key in csv_vibs:
    print(key)
    raw_data = csv_vibs[key]
    ax = plot_frequency_spectrum(n_disc, raw_data, key)

# Individual FFT    
n_disc = 300
key = '27741_raw data acceleration'                
raw_data = csv_vibs[key]
ax = plot_frequency_spectrum(n_disc, raw_data, key)    


###############################################################################
# getting velocity
###############################################################################        
# initialize an empty data frame
def initialize_df(raw_data):
    cols = raw_data.columns
    df = pd.DataFrame(columns=cols)
    return df

# get the velocity/displacement 
def get_velocity(raw_data):
    
    df = initialize_df(raw_data)
    
    n, _ = raw_data.shape
        
    for i in range(n-1):
    
        t1, t2 = raw_data.iloc[i:i+2, 0]
        
        a1 = raw_data.iloc[i,1:]
        a2 = raw_data.iloc[i+1,1:]
                
        mean_t = (t1 + t2)/2
        mean_a = (a1 + a2)/2
        delta_t = (t2 - t1)
        
        del_vel = mean_a*delta_t*9806   # 9806 conversion factor from g to mm/s2
        
        if i == 0:
            pre_vel = np.zeros(3)
        else:
            pre_vel = vel
            
        vel = pre_vel + del_vel
        vel['Timestamp'] = mean_t
            
        df = df.append(vel, ignore_index = True)
        
    return df

    
def removing_trend_by_differencing(df):
    df_diff = df.diff()
    df_diff.loc[:,'Timestamp'] = df.loc[:, 'Timestamp']
    df_diff = df_diff.dropna(axis = 0, how = 'any')
    return df_diff

def do_mean_centering(df):
    df_mean_cent = df - df.mean()
    df_mean_cent.loc[:,'Timestamp'] = df.loc[:, 'Timestamp']
    return df_mean_cent

def get_integ_df(raw_data):    
    df = get_velocity(raw_data)
    df1 = removing_trend_by_differencing(df)
    df2 = do_mean_centering(df1)
    return df2

def get_all_integ_df(dict_data, name):
    int_dfs = {}
    for key in dict_data:
        raw_data = dict_data[key]
        int_key = key.split('_')[0] + '_' + name
        int_df = get_integ_df(raw_data)
        int_dfs[int_key] = int_df
    return int_dfs
    
vel_dfs = get_all_integ_df(csv_vibs, 'velocity')
disp_dfs = get_all_integ_df(vel_dfs, 'displacement')


###############################################################################
# Final Analysis
###############################################################################
plot_all(csv_vibs)
plot_all(vel_dfs)
plot_all(disp_dfs)

###############################################################################
# velocity plot
###############################################################################
n_disc = 100
for key in vel_dfs:
    print(key)
    if key == '27741_velocity':
        n_disc = 250
    df = vel_dfs[key]
    plot_vib_data(key, df.iloc[n_disc:,:])
    

###############################################################################
# displacement plot
###############################################################################
n_disc = 100
for key in disp_dfs:
    print(key)
    if key == '27741_displacement':
        n_disc = 250
    df = disp_dfs[key]
    plot_vib_data(key, df.iloc[n_disc:,:])

###############################################################################
# FFT acceleration
###############################################################################
for key in csv_vibs:
    print(key)
    raw_data = csv_vibs[key]
    ax = plot_frequency_spectrum(n_disc, raw_data, key)





###############################################################################
# FFT velocity
###############################################################################





###############################################################################
# plotting the mean centered data
###############################################################################
df = vel_dfs.iloc[250:, :]
n_disc = 250
for i in range(1,4):
    plt.plot(vel_df.iloc[n_disc:,0], vel_df.iloc[n_disc:,i])
    plt.show()

###############################################################################
# Get FFT
###############################################################################
i = 1 
f, fft = get_fft(i, n_disc, vel_df)
n = len(f)
plt.plot(f[:n//2], np.abs(fft)[:n//2]*1/n)
plt.ylim(0,0.6)
plt.xlim(0.01, 800)








 
    
    
    
    
    
    
    
    
    
    