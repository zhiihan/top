import numpy as np
from scipy.special import ellipe, ellipk

import matplotlib.pyplot as plt
import math
import pandas as pd

phi = 0
t = 1
cos = np.linspace(0.001, 1, num = 400)

def var_energy(h, J, Delta, cos):
    
    arg1 = 1-Delta**2*cos**2/t**2
    return -h*np.sqrt(1 - cos**2)*np.cos(phi)-J*cos**2-(2*t*ellipe(arg1))/math.pi

def compute_data(h, J, Delta, cos):   
    
    data = var_energy(h, J, Delta, cos)
    # data = data[~np.isnan(data)]
    
    return {"J": J, 
           "Delta": Delta,
            "M^2": np.abs(cos[np.argmin(data)]),
            "h": h,
            "E_N": var_energy(h, J, Delta, cos[np.argmin(data)]),
            "t": t,
            "N": np.infty
           }