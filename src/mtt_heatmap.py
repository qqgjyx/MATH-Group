import numpy as np
import matplotlib.pyplot as plt
import torch
from dicom_perfusion_loader import DataLoader, GVFModel
import logging
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the logging level for all loggers to WARNING
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.WARNING)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize the DataLoader with your DICOM folder path
loader = DataLoader('SRS00013', device=device)

# Get the shape of the data
x, y, z, t = loader.get_shape()
logger.info(f"Data shape: {x}x{y}x{z}x{t}")

# Initialize the GVF model
gvf_model = GVFModel().to(device)

# Create an empty 3D tensor to store MTT values
mtt_map = torch.zeros((z, y, x), device=device)

# Option to process a subset of slices for testing
start_slice = 0
end_slice = z  # Change this to a smaller number for testing, e.g., 5

# Iterate through each slice
start_time = time.time()
error_count = 0
for z_idx in tqdm(range(start_slice, end_slice), desc="Processing slices"):
    # Get concentrations and time points for the entire slice
    concentrations = loader.get_concentration_slice(z_idx + 1)
    time_points = loader.get_time_points(z_idx + 1)
    
    # Reshape concentrations to (y*x, t)
    concentrations = concentrations.view(-1, t)
    
    # Fit the model and calculate metrics for the entire slice
    try:
        gvf_model.fit(time_points, concentrations)
        metrics = gvf_model.calculate_metrics()
        mtt_map[z_idx] = metrics["Mean Transit Time"].view(y, x)
    except Exception as e:
        logger.warning(f"Error processing slice {z_idx}: {str(e)}")
        error_count += 1
        mtt_map[z_idx] = torch.full((y, x), torch.nan, device=device)

end_time = time.time()
logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
logger.info(f"Total errors encountered: {error_count}")

# Move mtt_map to CPU for visualization and saving
mtt_map = mtt_map.cpu().numpy()

def display_mtt_slice(z_idx):
    plt.figure(figsize=(10, 8))
    plt.imshow(mtt_map[z_idx], cmap='hot', interpolation='nearest')
    plt.colorbar(label='Mean Transit Time (s)')
    plt.title(f'MTT Heat Map - Slice {z_idx + 1}')
    plt.axis('off')
    plt.show()

# Display MTT heat maps for a few slices
for z_idx in [start_slice, (start_slice + end_slice) // 2, end_slice - 1]:
    display_mtt_slice(z_idx)

# Save the MTT map as a NumPy array for future use
np.save('mtt_map.npy', mtt_map)

logger.info("MTT heat map generation complete.")

# Print statistics about the MTT map
valid_mtts = mtt_map[~np.isnan(mtt_map)]
logger.info(f"MTT statistics:")
logger.info(f"  Min MTT: {np.min(valid_mtts):.2f} s")
logger.info(f"  Max MTT: {np.max(valid_mtts):.2f} s")
logger.info(f"  Mean MTT: {np.mean(valid_mtts):.2f} s")
logger.info(f"  Median MTT: {np.median(valid_mtts):.2f} s")
logger.info(f"  Percentage of valid voxels: {100 * len(valid_mtts) / mtt_map.size:.2f}%")
