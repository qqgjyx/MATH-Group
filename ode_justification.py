#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import simps
import ipywidgets as widgets
from ipywidgets import interact

# Define Lorentzian function
def L(t, t0, A, Gamma):
    return (A/np.pi) * (Gamma / ((t - t0)**2 + Gamma**2))

# Define Ca and Cv as Lorentzian functions
def Ca(t):
    return L(t, 30, 5000, 4)

def Cv(t):
    return L(t, 36, 8000, 2.7)

# Precompute κ using Simpson's rule
time_grid_fine = np.linspace(0, 72, 1000)
integral_Ca = simps(Ca(time_grid_fine), time_grid_fine)
integral_Cv = simps(Cv(time_grid_fine), time_grid_fine)
kappa = integral_Ca / integral_Cv

# Define gamma distribution as the residue function R(t)
def gamma_dist(t, t1, k, theta):
    if t >= t1:
        return (1 / (gamma(k) * theta**k)) * (t - t1)**(k - 1) * np.exp(-(t - t1) / theta)
    else:
        return 0

# Precompute the time grid
time_grid = np.arange(0, 73, 1)

# Memoize CinValues to avoid recomputation
CinValuesCache = {t: (1/kappa) * ((1 - 0.35)/(1 - 0.45)) * simps([Ca(tau) * gamma_dist(t - tau, 4, 1, 2) for tau in time_grid_fine if tau <= t], time_grid_fine[time_grid_fine <= t]) for t in time_grid}

# Create an interpolation function for Cin
CinInterp = np.interp(time_grid, list(CinValuesCache.keys()), list(CinValuesCache.values()))

# Memoize CtValues to avoid recomputation
def CtValue(t, CinInterp):
    def integrand(k1):
        return (np.exp((20 * k1) / 96.15) * 20 * np.interp(k1, time_grid, CinInterp)) / 96.15
    
    return np.exp(-(20 * t) / 96.15) * simps([integrand(k1) for k1 in time_grid_fine if k1 <= t], time_grid_fine[time_grid_fine <= t])

CtValuesCache = {t: CtValue(t, CinInterp) for t in time_grid}

# Create an interpolation function for Ct
CtInterp = np.interp(time_grid, list(CtValuesCache.keys()), list(CtValuesCache.values()))

# Function to update and plot
def update_plot(t1, k1, sigma1, Q, V, HLV, HSV):
    def R(t):
        return gamma_dist(t, t1, k1, sigma1)
    
    CinValues = [(1/kappa) * ((1 - HLV) / (1 - HSV)) * simps([Ca(tau) * R(t - tau) for tau in time_grid_fine if tau <= t], time_grid_fine[time_grid_fine <= t]) for t in time_grid]
    CinInterp = np.interp(time_grid, time_grid, CinValues)
    CtValues = [np.exp(-(Q * t) / V) * simps([(np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V for k1 in time_grid_fine if k1 <= t], time_grid_fine[time_grid_fine <= t]) for t in time_grid]
    CtInterp = np.interp(time_grid, time_grid, CtValues)
    
    plt.figure(figsize=(12, 8))
    plt.plot(time_grid, [Ca(t) for t in time_grid], label='AIF (Ca)', color='red', linewidth=2)
    plt.plot(time_grid, [Cv(t) for t in time_grid], label='VOF (Cv)', color='blue', linewidth=2)
    plt.plot(time_grid, CtInterp, label='Tissue (Ct)', color='green', linewidth=2)
    plt.plot(time_grid, CinInterp, label='Cin', color='pink', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    plt.title('AIF, VOF, Tissue Concentration, and Convolved AIF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widgets
interact(update_plot,
         t1=widgets.FloatSlider(min=1, max=10, step=0.1, value=4, description='t1'),
         k1=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=1, description='k1'),
         sigma1=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=2, description='sigma1'),
         Q=widgets.FloatSlider(min=1, max=50, step=1, value=20, description='Q (cm^3/s/100g)'),
         V=widgets.FloatSlider(min=50, max=150, step=1, value=96.15, description='V (cm^3/100g)'),
         HLV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.45, description='HLV'),
         HSV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.35, description='HSV'))


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import ipywidgets as widgets
from ipywidgets import interact
from scipy.integrate import trapezoid

# Define Lorentzian function
def L(t, t0, A, Gamma):
    return (A / np.pi) * (Gamma / ((t - t0)**2 + Gamma**2))

# Define Ca and Cv as Lorentzian functions
def Ca(t):
    return L(t, 30, 5000, 4)

def Cv(t):
    return L(t, 36, 8000, 2.7)

# Precompute κ using quad
time_grid_fine = np.linspace(0, 72, 10)
integral_Ca, _ = quad(Ca, 0, 72)
integral_Cv, _ = quad(Cv, 0, 72)
kappa = integral_Ca / integral_Cv

# Define gamma distribution as the residue function R(t)
def gamma_dist(t, t1, k, theta):
    if t >= t1:
        return (1 / (gamma(k) * theta**k)) * (t - t1)**(k - 1) * np.exp(-(t - t1) / theta)
    else:
        return 0

# Precompute the time grid
time_grid = np.arange(0, 73, 1)

# Memoize CinValues to avoid recomputation
def Cin(t, t1, k, theta, HLV, HSV):
    def integrand(tau):
        return Ca(tau) * gamma_dist(t - tau, t1, k, theta)
    integral_value, _ = quad(integrand, 0, t)
    return (1 / kappa) * ((1 - HLV) / (1 - HSV)) * integral_value

CinValuesCache = {t: Cin(t, 4, 1, 2, 0.45, 0.35) for t in time_grid}
CinInterp = np.interp(time_grid, list(CinValuesCache.keys()), list(CinValuesCache.values()))

Q=20
V=96.15
def Ct(t, Cin):
    return np.exp(-(Q * t) / V) * (trapezoid([np.exp(Q * tao / V) * (Q / V) * Cin(tao) for tao in list(np.arange(0, t, 1))], list(np.arange(0, t, 1))))

CaInterp = np.interp(time_grid, time_grid, [Ca(t) for t in time_grid])
CValues = [np.exp(-(Q * t) / V) * quad(lambda k1: (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CaInterp)) / V, 0, t)[0] for t in time_grid]
Cinterp = np.interp(time_grid, time_grid, CValues)

# Memoize CtValues to avoid recomputation
def CtValue(t, CinInterp, Q, V):
    def integrand(k1):
        return (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V
    integral_value, _ = quad(integrand, 0, t)
    return np.exp(-(Q * t) / V) * integral_value

CtValuesCache = {t: CtValue(t, CinInterp, 20, 96.15) for t in time_grid}
CtInterp = np.interp(time_grid, list(CtValuesCache.keys()), list(CtValuesCache.values()))

# Function to update and plot
def update_plot(t1, k1, sigma1, Q, V, HLV, HSV):
    def R(t):
        return gamma_dist(t, t1, k1, sigma1)
    
    # CinValues = [(1 / kappa) * ((1 - HLV) / (1 - HSV)) * quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinValues = [quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinInterp = np.interp(time_grid, time_grid, CinValues)
    CtValues = [np.exp(-(Q * t) / V) * quad(lambda k1: (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V, 0, t)[0] for t in time_grid]
    CtInterp = np.interp(time_grid, time_grid, CtValues)
    
    plt.figure(figsize=(12, 8))
    plt.plot(time_grid, [Ca(t) for t in time_grid], label='Input', color='red', linewidth=2)
    # plt.plot(time_grid, [Cv(t) for t in time_grid], label='VOF (Cv)', color='blue', linewidth=2)
    # plt.plot(time_grid, CtInterp, label='Tissue (Ct)', color='green', linewidth=2)
    plt.plot(time_grid, CinInterp, label='Convolution', color='pink', linewidth=2, linestyle='--')
    plt.plot(time_grid, Cinterp, label='ODE Equation 1', color='green', linewidth=2, linestyle=':')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    plt.title('Connection between Convolution and ODE')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widgets
interact(update_plot,
         t1=widgets.FloatSlider(min=0, max=10, step=0.1, value=4, description='t1'),
         k1=widgets.FloatSlider(min=0, max=10, step=0.1, value=1, description='k1'),
         sigma1=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=2, description='sigma1'),
         Q=widgets.FloatSlider(min=1, max=50, step=1, value=20, description='Q (cm^3/s/100g)'),
         V=widgets.FloatSlider(min=50, max=150, step=1, value=96.15, description='V (cm^3/100g)'),
         HLV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.45, description='HLV'),
         HSV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.35, description='HSV'))


# In[2]:





# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import ipywidgets as widgets
from ipywidgets import interact
from scipy.integrate import trapezoid

# Define Lorentzian function
def L(t, t0, A, Gamma):
    return (A / np.pi) * (Gamma / ((t - t0)**2 + Gamma**2))

# Define Ca and Cv as Lorentzian functions
def Ca(t):
    return L(t, 30, 5000, 4)

def Cv(t):
    return L(t, 36, 8000, 2.7)

# Precompute κ using quad
time_grid_fine = np.linspace(0, 72, 10)
integral_Ca, _ = quad(Ca, 0, 72)
integral_Cv, _ = quad(Cv, 0, 72)
kappa = integral_Ca / integral_Cv

# Define gamma distribution as the residue function R(t)
def gamma_dist(t, t1, k, theta):
    if t >= t1:
        return (1 / (gamma(k) * theta**k)) * (t - t1)**(k - 1) * np.exp(-(t - t1) / theta)
    else:
        return 0

# Precompute the time grid
time_grid = np.arange(0, 73, 1)

# Memoize CinValues to avoid recomputation
def Cin(t, t1, k, theta, HLV, HSV):
    def integrand(tau):
        return Ca(tau) * gamma_dist(t - tau, t1, k, theta)
    integral_value, _ = quad(integrand, 0, t)
    return (1 / kappa) * ((1 - HLV) / (1 - HSV)) * integral_value

CinValuesCache = {t: Cin(t, 4, 1, 2, 0.45, 0.35) for t in time_grid}
CinInterp = np.interp(time_grid, list(CinValuesCache.keys()), list(CinValuesCache.values()))

# Memoize CtValues to avoid recomputation
def CtValue(t, CinInterp, Q, V):
    def integrand(k1):
        return (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V
    integral_value, _ = quad(integrand, 0, t)
    return np.exp(-(Q * t) / V) * integral_value

CtValuesCache = {t: CtValue(t, CinInterp, 20, 96.15) for t in time_grid}
CtInterp = np.interp(time_grid, list(CtValuesCache.keys()), list(CtValuesCache.values()))

# Function to update and plot
def update_plot(t1, k1, sigma1, Q, V, HLV, HSV):
    def R(t):
        return gamma_dist(t, t1, k1, sigma1)
    
    # CinValues = [(1 / kappa) * ((1 - HLV) / (1 - HSV)) * quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinValues = [quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinInterp = np.interp(time_grid, time_grid, CinValues)
    CtValues = [np.exp(-(Q * t) / V) * quad(lambda k1: (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V, 0, t)[0] for t in time_grid]
    CtInterp = np.interp(time_grid, time_grid, CtValues)
    
    plt.figure(figsize=(12, 8))
    plt.plot(time_grid, [Ca(t) for t in time_grid], label='AIF (Ca)', color='red', linewidth=2)
    plt.plot(time_grid, [Cv(t) for t in time_grid], label='VOF (Cv)', color='blue', linewidth=2)
    plt.plot(time_grid, CtInterp, label='Tissue (Ct)', color='green', linewidth=2)
    plt.plot(time_grid, CinInterp, label='Cin', color='pink', linewidth=2, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    plt.title('AIF, VOF, Tissue Concentration, and Convolved AIF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widgets
interact(update_plot,
         t1=widgets.FloatSlider(min=0, max=10, step=0.1, value=4, description='t1'),
         k1=widgets.FloatSlider(min=0, max=10, step=0.1, value=1, description='k1'),
         sigma1=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=2, description='sigma1'),
         Q=widgets.FloatSlider(min=1, max=50, step=1, value=20, description='Q (cm^3/s/100g)'),
         V=widgets.FloatSlider(min=50, max=150, step=1, value=96.15, description='V (cm^3/100g)'),
         HLV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.45, description='HLV'),
         HSV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.35, description='HSV'))


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
import ipywidgets as widgets
from ipywidgets import interact
from scipy.integrate import trapezoid

# Define square wave function
def square_wave(t, t0, A, width):
    return A if t0 - width/2 <= t <= t0 + width/2 else 0

# Define Ca and Cv as square wave functions
def Ca(t):
    return square_wave(t, 30, 5000, 8)  # Example parameters

def Cv(t):
    return square_wave(t, 36, 8000, 5.4)  # Example parameters

# Precompute κ using quad
time_grid_fine = np.linspace(0, 72, 10)
integral_Ca, _ = quad(Ca, 0, 72)
integral_Cv, _ = quad(Cv, 0, 72)
kappa = integral_Ca / integral_Cv

# Define gamma distribution as the residue function R(t)
def gamma_dist(t, t1, k, theta):
    if t >= t1:
        return (1 / (gamma(k) * theta**k)) * (t - t1)**(k - 1) * np.exp(-(t - t1) / theta)
    else:
        return 0

# Precompute the time grid
time_grid = np.arange(0, 73, 1)

# Memoize CinValues to avoid recomputation
def Cin(t, t1, k, theta, HLV, HSV):
    def integrand(tau):
        return Ca(tau) * gamma_dist(t - tau, t1, k, theta)
    integral_value, _ = quad(integrand, 0, t)
    return (1 / kappa) * ((1 - HLV) / (1 - HSV)) * integral_value

CinValuesCache = {t: Cin(t, 4, 1, 2, 0.45, 0.35) for t in time_grid}
CinInterp = np.interp(time_grid, list(CinValuesCache.keys()), list(CinValuesCache.values()))

# Memoize CtValues to avoid recomputation
def CtValue(t, CinInterp, Q, V):
    def integrand(k1):
        return (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V
    integral_value, _ = quad(integrand, 0, t)
    return np.exp(-(Q * t) / V) * integral_value

CtValuesCache = {t: CtValue(t, CinInterp, 20, 96.15) for t in time_grid}
CtInterp = np.interp(time_grid, list(CtValuesCache.keys()), list(CtValuesCache.values()))

# Function to update and plot
def update_plot(t1, k1, sigma1, Q, V, HLV, HSV):
    def R(t):
        return gamma_dist(t, t1, k1, sigma1)
    
    # CinValues = [(1 / kappa) * ((1 - HLV) / (1 - HSV)) * quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinValues = [quad(lambda tau: Ca(tau) * R(t - tau), 0, t)[0] for t in time_grid]
    CinInterp = np.interp(time_grid, time_grid, CinValues)
    CtValues = [np.exp(-(Q * t) / V) * quad(lambda k1: (np.exp((Q * k1) / V) * Q * np.interp(k1, time_grid, CinInterp)) / V, 0, t)[0] for t in time_grid]
    CtInterp = np.interp(time_grid, time_grid, CtValues)
    
    plt.figure(figsize=(12, 8))
    plt.plot(time_grid, [Ca(t) for t in time_grid], label='AIF (Ca)', color='red', linewidth=2)
    plt.plot(time_grid, [Cv(t) for t in time_grid], label='VOF (Cv)', color='blue', linewidth=2)
    plt.plot(time_grid, CtInterp, label='Tissue (Ct)', color='green', linewidth=2)
    plt.plot(time_grid, CinInterp, label='Cin', color='pink', linewidth=2, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration')
    plt.title('AIF, VOF, Tissue Concentration, and Convolved AIF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create interactive widgets
interact(update_plot,
         t1=widgets.FloatSlider(min=0, max=10, step=0.1, value=4, description='t1'),
         k1=widgets.FloatSlider(min=0, max=10, step=0.1, value=1, description='k1'),
         sigma1=widgets.FloatSlider(min=0.1, max=10, step=0.1, value=2, description='sigma1'),
         Q=widgets.FloatSlider(min=1, max=50, step=1, value=20, description='Q (cm^3/s/100g)'),
         V=widgets.FloatSlider(min=50, max=150, step=1, value=96.15, description='V (cm^3/100g)'),
         HLV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.45, description='HLV'),
         HSV=widgets.FloatSlider(min=0.1, max=1, step=0.01, value=0.35, description='HSV'))


# In[4]:




