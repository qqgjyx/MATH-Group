#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pydicom
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from scipy.ndimage import rotate
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
from IPython.display import display


# In[2]:


dcm_path = '/Users/juntangwang/Desktop/SRS00013/IMG00001.DCM'
folder_path = '/Users/juntangwang/Desktop/SRS00013/'  # Update this to the path of your DICOM folder


# In[3]:


def get_image_orientation(ds):
    # This function extracts the orientation of the image from the DICOM metadata
    return np.array(ds.ImageOrientationPatient)

def get_image_position(ds):
    # This function extracts the position of the image slice from the DICOM metadata
    return np.array(ds.ImagePositionPatient)

def sort_slices(dicom_files):
    # This function sorts the DICOM files based on their position and also retrieves the acquisition times
    sorted_slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        position = tuple(ds.ImagePositionPatient)
        acquisition_time = time_to_seconds(ds.AcquisitionTime)
        sorted_slices.append((position, acquisition_time, ds.pixel_array))
    # Sort by position, then by acquisition time if positions are the same
    sorted_slices.sort(key=lambda x: (x[0], x[1]))
    return sorted_slices


def rotate_image_to_standard_orientation(image, orientation):
    # This is a placeholder function. You'll need to implement the logic
    # to determine the correct rotation based on the orientation metadata.
    # Here's a simple example that assumes 'orientation' is the angle:
    return rotate(image, angle=orientation)


def time_to_seconds(t):
    """Converts a time string in HHMMSS.fff format to seconds."""
    hours, minutes, seconds = int(t[:2]), int(t[2:4]), float(t[4:])
    return 3600 * hours + 60 * minutes + seconds


def calculate_mtt(pixel_time_series, time_diffs):
    """Calculate MTT using time-weighted signal intensity."""
    weighted_intensity = pixel_time_series * time_diffs
    mtt = np.sum(weighted_intensity) / np.sum(time_diffs)
    return mtt



# Step 1: Data

# In[4]:


# Example for one DICOM file
ds = pydicom.dcmread(dcm_path)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()


# In[5]:


# Replace 'path_to_your_dicom_file' with the actual path to your DICOM file
dicom_file_path = dcm_path
ds = pydicom.dcmread(dicom_file_path)



# In[6]:


# Print all metadata
print("DICOM Metadata:\n")
for elem in ds:
    print(f"{elem.tag} {elem.description()}: {elem.value}")


# In[7]:


dicom_files = [f for f in os.listdir(folder_path) if f.endswith('.DCM')]

# Initialize a list to hold your image data
image_data = []

for file in dicom_files:
    file_path = os.path.join(folder_path, file)
    ds = pydicom.dcmread(file_path)
    
    # Preprocess the image as necessary. This is just a placeholder for any actual preprocessing you need to do.
    # For example: image = preprocess(ds.pixel_array)
    image_data.append(ds.pixel_array)
    
    


# In[8]:


image_data_np = np.array(image_data)
print(image_data_np.shape)
print(image_data_np.min(), image_data_np.max())


# In[9]:


# Assuming 'image_data' is already populated with images from the DICOM files
# Show one of the images before preprocessing
plt.figure(figsize=(6, 6))
plt.imshow(image_data[0], cmap='gray')  # Show the first image in the dataset
plt.title('Original Image')
plt.axis('off')
plt.show()


# In[10]:


# Example of applying Gaussian blur for noise reduction
image_data_blurred = [gaussian(image, sigma=1) for image in image_data]

# Example of applying contrast enhancement
image_data_enhanced = [equalize_adapthist(image) for image in image_data_blurred]

# Edge enhancement example is more complex as it usually applies to specific cases.


# In[11]:


# Apply Gaussian blur to the first image for noise reduction
image_blurred = image_data_blurred[0]

# Apply contrast enhancement to the blurred image
image_enhanced = image_data_enhanced[0]

# Show the blurred image
plt.figure(figsize=(6, 6))
plt.imshow(image_blurred, cmap='gray')
plt.title('Image after Gaussian Blur')
plt.axis('off')
plt.show()

# Show the enhanced image
plt.figure(figsize=(6, 6))
plt.imshow(image_enhanced, cmap='gray')
plt.title('Image after Contrast Enhancement')
plt.axis('off')
plt.show()


# In[12]:


# Example of extracting acquisition time from DICOM metadata
acquisition_times = []
image_orientations = []
image_positions = []

for file in dicom_files:
    file_path = os.path.join(folder_path, file)
    ds = pydicom.dcmread(file_path)
    
    # Extract acquisition time; note that you'll need to adjust 'AcquisitionTime' based on your DICOM files' metadata structure
    acquisition_time = ds.AcquisitionTime
    acquisition_times.append(acquisition_time)
    
    image_position = ds.ImagePositionPatient
    image_positions.append(image_position)
    
    image_orientation = ds.ImageOrientationPatient
    image_orientations.append(image_orientation)


# In[13]:


print(acquisition_times)


# In[14]:


print(image_positions)

points = image_positions
# Separate the points into x, y, and z coordinates for plotting
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]
z_coords = [point[2] for point in points]

# Create a new figure for the 3D plot
fig = plt.figure()

# Add a 3D subplot to the figure
ax = fig.add_subplot(111, projection='3d')

# Scatter the points in 3D space
ax.scatter(x_coords, y_coords, z_coords)

# Set labels for axes (optional)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()


# In[15]:


print(image_orientations)

points = image_orientations

# Convert each sublist to a tuple and add to a set to remove duplicates
unique_points_set = set(tuple(point) for point in points)

# Convert the set back to a list of lists
unique_points = [list(point) for point in unique_points_set]


# In[16]:


print(unique_points)


# In[17]:


acquisition_time_seconds = [time_to_seconds(t) for t in acquisition_times]
time_diffs = np.diff(acquisition_time_seconds)
time_diffs = np.append(time_diffs, time_diffs[-1])  # Append the last difference to maintain shape


# In[18]:


# Step 2: Assuming 'image_data_enhanced' is already aligned with 'acquisition_times'

# Convert 'image_data_enhanced' to a numpy array if it's not already
image_series = np.array(image_data_enhanced)  # This is now your 3D array [time, x, y]
acquisition_time_seconds = np.array(acquisition_time_seconds)
image_positions = np.array(image_positions)

print(image_series.shape)
print(acquisition_time_seconds.shape)
print(image_positions.shape)


# In[19]:


# Step 3: Calculate MTT
# The MTT calculation will depend on your specific methodology.
# The placeholder function 'calculate_mtt' assumes a very simplistic approach and will likely need to be replaced with your actual calculation method.

# Assuming image_series and time_diffs are correctly set up
mtt_map = np.zeros(image_series.shape[1:])  # [x, y]

for x in range(image_series.shape[1]):
    for y in range(image_series.shape[2]):
        pixel_time_series = image_series[:, x, y]
        mtt = calculate_mtt(pixel_time_series, time_diffs)
        mtt_map[x, y] = mtt

# Display the MTT map
plt.figure(figsize=(6, 6))
plt.imshow(mtt_map, cmap='hot')
plt.colorbar()
plt.title('Mean Transit Time (MTT) Map')
plt.axis('off')
plt.show()


# In[20]:


# Assuming 'image_series' is a 4D numpy array [time, z, x, y]
# where 'time' is the 4th dimension (different acquisition times)

# Sort your images temporally if needed. This assumes you have a list of acquisition times corresponding to each slice
sorted_indices = np.argsort(acquisition_time_seconds)
image_series_4d = image_series[sorted_indices]  # This reorders your 3D data along the time dimension

# Interactive visualization of 3D data over time
def explore_3dimage_time(time=0):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_series_4d[time, :, :], cmap='gray')  # Adjust the 15 here to show different slices
    plt.title(f'Time: {time}')
    plt.axis('off')
    plt.show()

# Create a slider to move through time
interact(explore_3dimage_time, time=IntSlider(min=0, max=image_series_4d.shape[0]-1, step=1, value=0))


# In[21]:


# Function to update the image based on the acquisition time and position
def update_image(time_index, position_index):
    plt.figure(figsize=(6, 6))
    # Extract the 2D image data for the given time and position index
    image_data = image_series[position_index] if position_index < len(image_series) else image_series[-1]
    plt.imshow(image_data, cmap='gray')  # Update to use the correct colormap
    plt.title(f'Time: {acquisition_time_seconds[time_index]}s, Position: {image_positions[position_index]}')
    plt.axis('off')
    plt.show()

# Function to get the closest index in the acquisition times array for a given time
def find_nearest_index(array, value):
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Widgets
time_slider = widgets.IntSlider(
    value=int(acquisition_time_seconds.min()),
    min=int(acquisition_time_seconds.min()),
    max=int(acquisition_time_seconds.max()),
    step=1,
    description='Time (s):',
    continuous_update=False
)

position_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(image_positions) - 1,
    step=1,
    description='Position Index:',
    continuous_update=False
)

# Display the widgets
display(time_slider)
display(position_slider)

# Callback function to update image when any slider's value changes
def on_change(change):
    time_index = find_nearest_index(acquisition_time_seconds, time_slider.value)
    update_image(time_index, position_slider.value)

# Observe changes in the slider values
time_slider.observe(on_change, names='value')
position_slider.observe(on_change, names='value')

# Initially display the first image
update_image(0, 0)


# In[ ]:




