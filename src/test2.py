import numpy as np
import matplotlib.pyplot as plt
from dicom_perfusion_loader import GVFModel
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Generate sample data
t = np.linspace(0, 100, 100)
A, t0, alpha, beta = 100, 10, 2, 5
y = GVFModel.gamma_variate(t, A, t0, alpha, beta) + np.random.normal(0, 5, 100)

# Ensure no negative values in y (which could cause issues when taking reciprocal)
y = np.maximum(y, 1e-10)

logger.debug(f"Generated data - t shape: {t.shape}, y shape: {y.shape}")
logger.debug(f"t range: [{np.min(t)}, {np.max(t)}], y range: [{np.min(y)}, {np.max(y)}]")

# Create and fit the model
model = GVFModel()
try:
    fitted_params = model.fit(t, y)
    print("True parameters:", (A, t0, alpha, beta))
    print("Fitted parameters:", fitted_params)

    # Predict using the fitted model
    predicted_y = model.predict(t)

    # Calculate metrics
    metrics = model.calculate_metrics()
    print("\nCalculated metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'o', label='Original data')
    plt.plot(t, predicted_y, '-', label='Fitted GVF')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('Gamma Variate Function Fit')
    plt.show()

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error(traceback.format_exc())

# Print some statistics about the data
print("\nData statistics:")
print(f"Min value: {np.min(y)}")
print(f"Max value: {np.max(y)}")
print(f"Mean value: {np.mean(y)}")
print(f"Number of infinite values: {np.sum(np.isinf(y))}")
print(f"Number of NaN values: {np.sum(np.isnan(y))}")

# Additional debugging: print out the first few values of t and y
print("\nFirst few values of t:")
print(t[:10])
print("\nFirst few values of y:")
print(y[:10])
