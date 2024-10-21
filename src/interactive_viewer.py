# %matplotlib inline
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider
from dicom_perfusion_loader import DataLoader
import numpy as np

# Initialize the DataLoader with your DICOM folder path
loader = DataLoader('path/to/your/dicom/folder')

# Get the shape of the data
x, y, z, t = loader.get_shape()

def view_slice(z, t):
    # Get the slice data for the given z
    slice_data = loader.get_slice(z)
    
    # Display the image for the given time point t
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data[t], cmap='gray')
    plt.title(f'Slice {z}/{z}, Time point {t+1}/{t}')
    plt.axis('off')
    plt.show()

# Create interactive widget for slice viewing
interact(
    view_slice,
    z=IntSlider(min=1, max=z, step=1, value=1, description='Slice:'),
    t=IntSlider(min=0, max=t-1, step=1, value=0, description='Time:')
)

# Function to plot intensity-time curve
def plot_intensity_curve(x, y, z):
    concentrations = loader.get_concentration(x, y, z)
    time_points = loader.get_time_points(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, concentrations, '-o')
    plt.title(f'Intensity-Time Curve at (x={x}, y={y}, z={z})')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    plt.grid(True)
    plt.show()

# Create interactive widget for intensity-time curve
interact(
    plot_intensity_curve,
    x=IntSlider(min=0, max=x-1, step=1, value=x//2, description='X:'),
    y=IntSlider(min=0, max=y-1, step=1, value=y//2, description='Y:'),
    z=IntSlider(min=1, max=z, step=1, value=z//2, description='Slice:')
)

# For demonstration purposes, let's display a few example slices
for z_val in [1, z//2, z]:
    for t_val in [0, t//2, t-1]:
        view_slice(z_val, t_val)

# Plot intensity-time curve for a specific voxel
plot_intensity_curve(x//2, y//2, z//2)
