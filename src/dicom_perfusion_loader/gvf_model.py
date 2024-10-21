import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GVFModel:
    def __init__(self):
        self.params = None

    @staticmethod
    def gamma_variate(t, A, t0, alpha, beta):
        """
        Gamma Variate Function with improved numerical stability
        """
        # Ensure t0 is not greater than t
        t0 = np.minimum(t0, np.min(t))
        
        # Avoid negative values in the power function
        power_term = np.maximum(t - t0, 0) ** alpha
        
        # Avoid division by zero or very small numbers
        exp_term = np.exp(-np.maximum(t - t0, 0) / np.maximum(beta, 1e-10))
        
        result = A * power_term * exp_term
        
        # Replace any potential inf or nan values with 0
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result

    def fit(self, t, y):
        """
        Fit the Gamma Variate Function to the given data
        
        Parameters:
        t : array-like
            Time points
        y : array-like
            Concentration values
        
        Returns:
        tuple: Optimal parameters (A, t0, alpha, beta)
        """
        logger.debug(f"Initial t shape: {t.shape}, y shape: {y.shape}")
        logger.debug(f"Initial t range: [{np.min(t)}, {np.max(t)}], y range: [{np.min(y)}, {np.max(y)}]")
        
        # Remove any infinite or NaN values
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]
        
        logger.debug(f"After removing inf/nan, t shape: {t.shape}, y shape: {y.shape}")
        logger.debug(f"After removing inf/nan, t range: [{np.min(t)}, {np.max(t)}], y range: [{np.min(y)}, {np.max(y)}]")

        if len(t) < 4:  # We need at least 4 points to fit 4 parameters
            raise ValueError("Not enough valid data points to fit the model")

        # Initial guess for parameters
        A_guess = np.max(y)
        t0_guess = t[np.argmax(y)] - 1
        alpha_guess = 1.0
        beta_guess = 1.0
        
        # Fit the function
        try:
            popt, _ = curve_fit(self.gamma_variate, t, y, 
                                p0=[A_guess, t0_guess, alpha_guess, beta_guess],
                                bounds=([0, np.min(t), 0, 0], [np.inf, np.max(t), np.inf, np.inf]),
                                maxfev=10000,  # Increase max function evaluations
                                method='trf')  # Try the Trust Region Reflective algorithm
        except RuntimeError as e:
            logger.error(f"Curve fitting failed: {str(e)}")
            logger.debug(f"Using initial guesses: A={A_guess}, t0={t0_guess}, alpha={alpha_guess}, beta={beta_guess}")
            popt = [A_guess, t0_guess, alpha_guess, beta_guess]
        
        self.params = popt
        logger.info(f"Fitted parameters: A={popt[0]}, t0={popt[1]}, alpha={popt[2]}, beta={popt[3]}")
        return popt

    def predict(self, t):
        """
        Predict concentration values using the fitted parameters
        
        Parameters:
        t : array-like
            Time points
        
        Returns:
        array-like: Predicted concentration values
        """
        if self.params is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.gamma_variate(t, *self.params)

    def calculate_metrics(self):
        """
        Calculate perfusion metrics based on the fitted GVF
        
        Returns:
        dict: Dictionary containing perfusion metrics
        """
        if self.params is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        A, t0, alpha, beta = self.params
        
        # Time to peak
        TTP = t0 + alpha * beta
        
        # Maximum concentration
        Cmax = self.gamma_variate(TTP, A, t0, alpha, beta)
        
        # Mean transit time
        MTT = alpha * beta
        
        # Area under the curve
        AUC = A * beta * gamma(alpha + 1)
        
        return {
            "Time to Peak": TTP,
            "Maximum Concentration": Cmax,
            "Mean Transit Time": MTT,
            "Area Under Curve": AUC
        }
