"""Load the DEAP Monte Carlo simulation data from the Joblib file"""
# Created by Brendon Matusch, August 2018

import os
from typing import List, Tuple

import numpy as np
from sklearn.externals import joblib


def load_deap_data():
    """Load and return the DEAP Monte Carlo simulation data"""
    # Use Joblib to load the binary file, and return it directly
    with open(os.path.expanduser('~/deap_data.pkl'), 'rb') as joblib_file:
        return joblib.load(joblib_file)
