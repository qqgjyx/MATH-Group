import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
import logging
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GVFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(100.0))
        self.t0 = nn.Parameter(torch.tensor(10.0))
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(5.0))

    def forward(self, t):
        t0 = torch.min(self.t0, torch.min(t))
        power_term = torch.maximum(t - t0, torch.tensor(0.0)) ** self.alpha
        exp_term = torch.exp(-torch.maximum(t - t0, torch.tensor(0.0)) / torch.maximum(self.beta, torch.tensor(1e-10)))
        result = self.A * power_term * exp_term
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, t, y):
        optimizer = optim.Adam(self.parameters())
        loss_fn = nn.MSELoss()
        
        for _ in range(1000):  # You might want to adjust the number of iterations
            optimizer.zero_grad()
            y_pred = self(t)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    def calculate_metrics(self):
        TTP = self.t0 + self.alpha * self.beta
        Cmax = self(TTP)
        MTT = self.alpha * self.beta
        # Note: AUC calculation might need to be approximated or calculated differently in PyTorch
        
        return {
            "Time to Peak": TTP,
            "Maximum Concentration": Cmax,
            "Mean Transit Time": MTT,
        }
