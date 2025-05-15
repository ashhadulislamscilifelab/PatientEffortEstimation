import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

from scipy.optimize import least_squares
import numpy as np
import random

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
from sklearn.metrics import r2_score




# 𝐹𝑙𝑜𝑤(𝑡)=𝑃𝑎𝑤(𝑡)/𝑅−𝑉𝑜𝑙𝑢𝑚𝑒(𝑡)∗𝐸/𝑅+𝐸𝑑𝑖(𝑡)∗𝑁𝑀𝐸/𝑅−𝑃0(𝑏𝑛)/𝑅
# Flow
## P0 individualized
def residuals_f(params, Flow, Press, Vol, Edi, breath_indices):
    R, E, NME = params[:3]
    P0_values = params[3:]
    
    # Map each time point to its corresponding breath's P0
    P0_t = np.array([P0_values[bn] for bn in breath_indices])

    Flow_est = (Press - Vol * E + Edi * NME - P0_t) / R
    return Flow - Flow_est



def estimateFlow(params, Flow, Press, Vol, Edi, breath_indices):
    """
    Compute estimated Flow based on learned parameters.
    
    Parameters:
    - params: array, where params[0:3] = R, E, NME and params[3:] = P0 per breath
    - Flow, Vol, Press, Edi: 1D numpy arrays (full time series)
    - breath_indices: 1D array mapping each time point to breath index (0, 1, ..., num_breaths-1)
    
    Returns:
    - Flow_est: estimated flow time series (same shape as Flow)
    """
    R, E, NME = params[:3]
    P0_values = params[3:]
    P0_t = np.array([P0_values[bn] for bn in breath_indices])
    
    Flow_est = (Press - Vol * E + Edi * NME - P0_t) / R
    return Flow_est    


## P0 shared
def residuals_f_singleP0(params, Flow, Press, Vol, Edi):
    R, E, NME, P0 = params
    Flow_est = (Press - Vol * E + Edi * NME - P0) / R
    return Flow - Flow_est

def estimateFlow_singleP0(params, Flow, Press, Vol, Edi):
    R, E, NME, P0 = params
    Flow_est = (Press - Vol * E + Edi * NME - P0) / R
    return Flow_est



# 𝑃𝑎𝑤(𝑡)=𝐹𝑙𝑜𝑤(𝑡)∗𝑅+𝑉𝑜𝑙𝑢𝑚𝑒(𝑡)∗𝐸−𝐸𝑑𝑖(𝑡)∗𝑁𝑀𝐸+𝑃0(𝑏𝑛)
## Press
## P0 individualized
def residuals_p(params, Flow, Press, Vol, Edi, breath_indices):
    R, E, NME = params[:3]
    P0_values = params[3:]
    P0_t = np.array([P0_values[bn] for bn in breath_indices])
    Press_est = Flow * R + Vol * E - Edi * NME + P0_t
    return Press - Press_est

def estimatePress(params, Flow, Press, Vol, Edi, breath_indices):
    R, E, NME = params[:3]
    P0_values = params[3:]
    P0_t = np.array([P0_values[bn] for bn in breath_indices])
    return Flow * R + Vol * E - Edi * NME + P0_t

## P0 shared


def residuals_p_singleP0(params, Flow, Press, Vol, Edi):
    R, E, NME, P0 = params
    Press_est = Flow * R + Vol * E - Edi * NME + P0
    return Press - Press_est

def estimatePress_singleP0(params, Flow, Press, Vol, Edi):
    R, E, NME, P0 = params
    return Flow * R + Vol * E - Edi * NME + P0


# 𝑉𝑜𝑙𝑢𝑚𝑒(𝑡)=𝑃𝑎𝑤(𝑡)/𝐸−𝐹𝑙𝑜𝑤(𝑡)∗𝑅/𝐸+𝐸𝑑𝑖(𝑡)∗𝑁𝑀𝐸/𝐸−𝑃0(𝑏𝑛)/𝐸
# Vol
## P0 individualized
def residuals_v(params, Flow, Press, Vol, Edi, breath_indices):
    R, E, NME = params[:3]
    P0_values = params[3:]
    P0_t = np.array([P0_values[bn] for bn in breath_indices])
    Vol_est = (Press - Flow * R + Edi * NME - P0_t) / E
    return Vol - Vol_est

def estimateVol(params, Flow, Press, Vol, Edi, breath_indices):
    R, E, NME = params[:3]
    P0_values = params[3:]
    P0_t = np.array([P0_values[bn] for bn in breath_indices])
    return (Press - Flow * R + Edi * NME - P0_t) / E
## P0 shared
def residuals_v_singleP0(params, Flow, Press, Vol, Edi,):
    R, E, NME, P0 = params
    Vol_est = (Press - Flow * R + Edi * NME - P0) / E
    return Vol - Vol_est

def estimateVol_singleP0(params, Flow, Press, Vol, Edi,):
    R, E, NME, P0 = params
    return (Press - Flow * R + Edi * NME - P0) / E
