#!/usr/bin/env python2
"""Load the DEAP Monte Carlo simulation data from ROOT files and save in a portable scikit-learn Joblib format"""
# Created by Brendon Matusch, August 2018
# Written in Python 2 because the DEAP-specific RAT distribution required for loading the relevant files binds to Python 2

import os

# Import RAT even though it is not used directly, because it modifies ROOT
import rat
import ROOT

# Load the data file and get the main tree
data_file = ROOT.TFile(os.path.expanduser('~/PB_000000_analyzed_0100.root'))
tree = data_file.Get('T')
tree.ls()
tree.Show(10)
a = tree.Draw('mc')
#x = tree[10]
#m = x.mc
#b = m.GetMCEventID()
