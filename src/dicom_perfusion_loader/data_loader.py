import os
import pydicom
import numpy as np
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from .utils import time_to_seconds, intensity_to_concentration

class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_data = []
        self.acquisition_times = []
        self.image_positions = []
        self.acquisition_numbers = []
        self.instance_numbers = []
        self.shape = None
        self.load_data()
        self.sort_data()
        self.set_shape()

    def load_data(self):
        dicom_files = [f for f in os.listdir(self.folder_path) if f.endswith('.DCM')]
        for file in dicom_files:
            file_path = os.path.join(self.folder_path, file)
            ds = pydicom.dcmread(file_path)
            self.image_data.append(equalize_adapthist(gaussian(ds.pixel_array, sigma=1)))
            self.acquisition_times.append(time_to_seconds(ds.AcquisitionTime))
            self.image_positions.append(ds.ImagePositionPatient)
            self.acquisition_numbers.append(ds.AcquisitionNumber)
            self.instance_numbers.append(ds.InstanceNumber)

    def sort_data(self):
        combined = list(zip(self.image_data, self.image_positions, self.acquisition_times, 
                            self.acquisition_numbers, self.instance_numbers))
        combined.sort(key=lambda x: (x[1][2], x[2]))  # Sort by z-position first, then by time
        (self.image_data, self.image_positions, self.acquisition_times, 
         self.acquisition_numbers, self.instance_numbers) = zip(*combined)
        self.image_data = np.array(self.image_data)

    def set_shape(self):
        if len(self.image_data) > 0:
            t = len(self.image_data)
            z = len(set([pos[2] for pos in self.image_positions]))
            y, x = self.image_data[0].shape
            self.shape = (x, y, z, t // z)
            self.image_data = self.image_data.reshape(self.shape[2], self.shape[3], self.shape[1], self.shape[0])

    def get_shape(self):
        return self.shape

    def get_slice(self, z):
        return self.image_data[z-1]  # z is 1-indexed, so we subtract 1

    def get_concentration(self, x, y, z):
        slice_data = self.get_slice(z)
        intensities = slice_data[:, y, x]  # Note the order: [t, y, x]
        return intensity_to_concentration(intensities)

    def get_time_points(self, z):
        x, y, total_z, t = self.shape
        start_idx = (z - 1) * t
        end_idx = z * t
        slice_times = self.acquisition_times[start_idx:end_idx]
        return [t - slice_times[0] for t in slice_times]

    # Add more methods as needed
