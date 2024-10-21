#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import deconvolve
from scipy.signal import convolve
from scipy.special import gamma
from scipy.optimize import minimize
import pandas as pd


# In[2]:


dcm_path = 'SRS00013/IMG00001.DCM'
folder_path = 'SRS00013/'  # Update this to the path of your DICOM folder


# In[3]:


def time_to_seconds(t):
    """Converts a time string in HHMMSS.fff format to seconds."""
    hours, minutes, seconds = int(t[:2]), int(t[2:4]), float(t[4:])
    return 3600 * hours + 60 * minutes + seconds

dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.DCM')]

# Initialize a list to hold your image data
image_data = []
acquisition_times = []
image_positions = []

acquisition_numbers =[]
instance_numbers = []

for file in dicom_files:
    file_path = os.path.join(folder_path, file)
    ds = pydicom.dcmread(file_path)
    
    # Preprocess the image as necessary. This is just a placeholder for any actual preprocessing you need to do.
    # For example: image = preprocess(ds.pixel_array)
    image_data.append(equalize_adapthist(gaussian(ds.pixel_array, sigma=1)))
    # image_data.append(ds.pixel_array)
    
    # Extract acquisition time; note that you'll need to adjust 'AcquisitionTime' based on your DICOM files' metadata structure
    acquisition_times.append(time_to_seconds(ds.AcquisitionTime))
    image_positions.append(ds.ImagePositionPatient)
    
    acquisition_numbers.append(ds.AcquisitionNumber)
    instance_numbers.append(ds.InstanceNumber)
    
image_data = np.array(image_data)
image_positions = np.array(image_positions)
acquisition_times = np.array(acquisition_times)

# Assuming these are your original lists:
# image_data = [np.array(...) for _ in range(1000)]  # Each a 256x256 numpy array
# image_position = [np.array(...) for _ in range(1000)]  # Each a 1x3 numpy array
# acquisition_time = [np.random.rand() for _ in range(1000)]  # Each a random float

# Step 1: Combine the lists into a single list of tuples
combined = list(zip(image_data, image_positions, acquisition_times, acquisition_numbers, instance_numbers))

# combined.sort(key=lambda x: x[-1])

combined.sort(key=lambda x: x[2])
# Step 2: Sort the combined list by the image_position (assuming it's the second element of the tuple)
combined.sort(key=lambda x: x[1][1])  # Adjust the lambda function if sorting criteria are different

# Step 3: Separate the combined list back into three lists
image_data_sorted, image_position_sorted, acquisition_time_sorted, acquisition_numbers_sorted, instance_numbers_sorted = zip(*combined)

# If you need the results to be in list format instead of tuples (especially for the image data), you can convert them
# image_data_sorted = list(equalize_adapthist(gaussian(np.array(image_data_sorted) , sigma=1)))
image_data_sorted = list(image_data_sorted)
image_position_sorted = list(image_position_sorted)
acquisition_time_sorted = list(acquisition_time_sorted)
acquisition_numbers_sorted = list(acquisition_numbers_sorted)
instance_numbers_sorted = list(instance_numbers_sorted)


# In[4]:


def intensity_to_concentration(intensity, baseline=None):
    
    if baseline is None:
        baseline = np.average(intensity[6:8]) #TODO: Baseline modeling
    
    baseline = baseline + np.min(baseline)
    
    # Placeholder values for demonstration
    te = 32  # Echo time in milliseconds (ms)
    s0 = baseline  # Baseline signal intensity before contrast injection
    st = intensity  # Signal intensity at different times after contrast injection

    # Calculating contrast agent concentration using the provided formula
    # ct = (1/te) * np.log(st/s0)
    ct = 1/st
    return ct

def plt_point(x,y,z,view=False):
    times = acquisition_time_sorted[50*(z-1):50*z]-acquisition_time_sorted[50*(z-1)]
    concentrations = intensity_to_concentration(image_data_sorted[50*(z-1):50*z,x,y])
    if view:
        print("InstanceNumbers: ")
        print(instance_numbers_sorted[50*(z-1):50*z])
        print("AcquisitionNumbers: ")
        print(acquisition_numbers_sorted[50*(z-1):50*z])
        print("unique image_positions: ")
        print(np.unique(image_position_sorted[50*(z-1):50*z]))
    
    # Assuming 'concentration' and 'time_points' are your data arrays
    plt.figure(figsize=(10, 6))
    plt.plot(times, concentrations, '-o', label='Concentration over Time')
    plt.title(f'Concentration vs. Time at <{x},{y},{z}>*\n*: not exact position, x,y is pixel index & z is slice index')
    plt.xlabel('Time Point (s)')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Interactive visualization of 3D data over time
def explore_3dimage_time(i=0,z=0,x=0,y=0,vis=False,aif=False):
    if vis:
        i = (z-1)*50+i
        plt.figure(figsize=(10, 5),facecolor='k')
        plt.imshow(image_data_sorted[i, :, :], cmap='gray')  # Adjust the 15 here to show different slices
        plt.title(f'#: {instance_numbers_sorted[i]} Position: {image_position_sorted[i]} \n T#: {acquisition_numbers_sorted[i]} Time: {acquisition_time_sorted[i]}',color='blue')
        plt.axis('off')
        plt.scatter(x, y, color='red', s=10)  # Highlight the point
        plt.tight_layout()
        plt.show()
    if aif: 
        plt_point(x,y,z)


# In[5]:


# Assuming 'image_series' is a 4D numpy array [time, z, x, y]
# where 'time' is the 4th dimension (different acquisition times)
image_data_sorted = np.array(image_data_sorted)

# Create a slider to move through time
interact(explore_3dimage_time, 
         i=IntSlider(min=0, max=49, step=1, value=0), 
         z=IntSlider(min=1,max=20,step=1, value=1),
         x=IntSlider(min=0, max=255, step=1),
         y=IntSlider(min=0, max=255, step=1),
         vis=True,
         aif=True)


# In[6]:


def smooth_data_gaussian(data, sigma=1):
    """
    Smooths the data using a Gaussian filter.

    :param data: The input data to smooth.
    :param sigma: The standard deviation for the Gaussian kernel.
    :return: The smoothed data.
    """
    return gaussian_filter1d(data, sigma)

def plt_pointss(xyztuples=None, view=False, smooth=False, legend_on=False):
    if xyztuples is None:
        xyztuples = [(0, 0, 0)]
    plt.figure(figsize=(5, 6))
    
    for i, (x, y, z) in enumerate(xyztuples):
        times = acquisition_time_sorted[50*(z-1):50*z] - acquisition_time_sorted[50*(z-1)]
        concentrations = intensity_to_concentration(image_data_sorted[50*(z-1):50*z,x,y])
        
        # Smooth the concentration data using Gaussian filter
        if smooth:
            smoothed_concentrations = smooth_data_gaussian(concentrations)
        else:
            smoothed_concentrations = concentrations.copy()

        if view:
            print(f"Data for {x},{y},{z}")
            print("InstanceNumbers: ")
            print(instance_numbers_sorted[50*(z-1):50*z])
            print("AcquisitionNumbers: ")
            print(acquisition_numbers_sorted[50*(z-1):50*z])
            print("unique image_positions: ")
            print(np.unique(image_position_sorted[50*(z-1):50*z]))
            print()  # Print a newline for readability
        
        plt.plot(times, smoothed_concentrations, '-o', label=f'<{x},{y},{z}>')
    
    plt.title('Smoothed Concentration vs. Time for Multiple Points\n*: not exact position, x,y is pixel index & z is slice index')
    plt.xlabel('Time Point (s)')
    plt.ylabel('Concentration')
    if legend_on:    
        plt.legend()
    plt.grid(True)
    plt.show()


# In[7]:


xyztuples = [  
    (171,110,9),
    
    # (173,103,13),
    # (173,104,13),
    (171,108,13),

    (124,195,17)
    ]
plt_pointss(xyztuples,smooth=True, legend_on=True)


# In[8]:


def test_around(x0,y0,z0,r=5,s=False, l=False):
    xyztuples=[]
    for x in range(x0-r,x0+r+1):
        for y in range(y0-r,y0+r+1):
            xyztuples.append((x,y,z0))
    plt_pointss(xyztuples, smooth=s, legend_on=l)

r = 3

interact(test_around,
         z0=IntSlider(min=1,max=20,step=1, value=1),
         x0=IntSlider(min=r, max=255-r, step=1),
         y0=IntSlider(min=r, max=255-r, step=1),
         r=r,
         l=False
         )


# In[10]:


def read_point4time(x,y,z):
    """Read time points for a given slice."""
    return acquisition_time_sorted[50*(z-1):50*z] - acquisition_time_sorted[50*(z-1)]

def read_point4concentration(x, y, z):
    """Read concentration data for a given point."""
    return intensity_to_concentration(image_data_sorted[50*(z-1):50*z, x, y])  


# In[11]:


def calculate_r_f(aif_concentration, tissue_concentration):
    """
    Calculate R(t) and F for multiple tissue points.
    
    :param aif_concentration: Tuple containing the concentration of the AIF pixel (x, y, z).
    :param tissue_concentration: Tuple containing the concentration of the tissue pixel (x, y, z).
    :return: A dictionary with tissue point coordinates as keys and (R_t, F) tuples as values.
    """
    # Define the hematocrit correction factor kH
    Ha = 0.45  # Typical arterial hematocrit
    Ht = 0.25  # Typical tissue hematocrit        
    kH = (1 - Ha) / (1 - Ht)
        
    # Apply deconvolution
    R_t, remainder = deconvolve(tissue_concentration, aif_concentration)
    F = (1 + remainder / convolve(aif_concentration, R_t)) *kH
            
    return R_t, F

# Example usage
# aif_pixel = (169, 99, 9)  # AIF pixel coordinates
# tissue_pixel = (173, 101,10)

aif_pixel = (171,110,9)
# tissue_pixel = (171,108,13)
tissue_pixel = (124,195,17)

R_t, F_t= calculate_r_f(read_point4concentration(*aif_pixel),read_point4concentration(*tissue_pixel))

print(f"R(t):", R_t)
print(f"Tissue blood flow (F):", F_t.shape)

time = read_point4time(*tissue_pixel)

# Define the hematocrit correction factor kH
Ha = 0.45  # Typical arterial hematocrit
Ht = 0.25  # Typical tissue hematocrit        
kH = (1 - Ha) / (1 - Ht)

AIFa_t = read_point4concentration(*aif_pixel)
C_t = read_point4concentration(*tissue_pixel)
Cr = np.convolve(AIFa_t, R_t, mode='full')

plt.figure(figsize=(7, 6))
plt.plot(time, AIFa_t, label='AIF')
plt.plot(time, C_t, label='VOF')
# plt.hlines(R_t, xmin=0,xmax=max(time),label='R(t)', linestyle='--')
plt.plot(time, Cr + 0.05, label='Recovered VOF\nOffset0.05', linestyle='-.')
plt.plot(time, Cr * F_t / kH + 0.05, label='Recovered VOF * F / kH\nOffset0.05', linestyle='-.')
plt.xlabel('Scan Time (s)')
plt.ylabel('Concentration (arb. units)')
plt.legend()
plt.title('AIF: AIFa(t), VOF: C(t) and Recovered')
plt.grid(True)

plt.xlim(0,77)

# plt.savefig('AIF_VOF_Recover.png', facecolor='black', transparent=True)
plt.show()


# In[ ]:




