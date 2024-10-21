import pytest
import numpy as np
from dicom_perfusion_loader import GVFModel

@pytest.fixture
def sample_data():
    # Generate sample time points and concentration values
    t = np.linspace(0, 100, 100)
    A, t0, alpha, beta = 100, 10, 2, 5
    y = GVFModel.gamma_variate(t, A, t0, alpha, beta) + np.random.normal(0, 5, 100)
    return t, y, (A, t0, alpha, beta)

def test_gvf_model_fit(sample_data):
    t, y, true_params = sample_data
    model = GVFModel()
    fitted_params = model.fit(t, y)
    
    # Check if fitted parameters are close to true parameters
    assert np.allclose(fitted_params, true_params, rtol=0.2)

def test_gvf_model_predict(sample_data):
    t, y, _ = sample_data
    model = GVFModel()
    model.fit(t, y)
    
    predicted_y = model.predict(t)
    
    # Check if predicted values are close to original values
    assert np.allclose(predicted_y, y, rtol=0.2)

def test_gvf_model_calculate_metrics(sample_data):
    t, y, _ = sample_data
    model = GVFModel()
    model.fit(t, y)
    
    metrics = model.calculate_metrics()
    
    # Check if all expected metrics are present
    expected_metrics = ["Time to Peak", "Maximum Concentration", "Mean Transit Time", "Area Under Curve"]
    for metric in expected_metrics:
        assert metric in metrics
    
    # Check if metrics have reasonable values
    assert 0 < metrics["Time to Peak"] < t[-1]
    assert 0 < metrics["Maximum Concentration"] < np.max(y) * 1.2
    assert 0 < metrics["Mean Transit Time"] < t[-1]
    assert 0 < metrics["Area Under Curve"] < np.sum(y) * (t[1] - t[0]) * 1.2

def test_gvf_model_errors():
    model = GVFModel()
    
    # Test error when predicting before fitting
    with pytest.raises(ValueError):
        model.predict(np.linspace(0, 100, 100))
    
    # Test error when calculating metrics before fitting
    with pytest.raises(ValueError):
        model.calculate_metrics()

def test_gvf_model_gamma_variate():
    t = np.linspace(0, 100, 100)
    A, t0, alpha, beta = 100, 10, 2, 5
    y = GVFModel.gamma_variate(t, A, t0, alpha, beta)
    
    # Check if function returns expected shape
    assert y.shape == t.shape
    
    # Check if function is zero before t0
    assert np.all(y[t < t0] == 0)
    
    # Check if function reaches maximum at expected time
    expected_peak_time = t0 + alpha * beta
    actual_peak_time = t[np.argmax(y)]
    assert np.isclose(actual_peak_time, expected_peak_time, rtol=0.1)
