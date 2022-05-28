# maths
import pandas as pd
import numpy as np
import seaborn as sns
from numpy.fft import fft
from scipy import stats

# plotting
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from IPython.display import HTML
import altair as alt

# noise reduction test
# import noisereduce as nr
# import padasip as pa

# os
from time import time
import os
from pathlib import Path

# own libraries
from nfc_signal_offline import *
from device import *
from nfc_signal_help import *
from nfc_signal_demodulation_stats import *

np.random.seed(52102) # always use the same random seed to make results comparable
