#!/usr/bin/env python3
"""A script for analyzing and printing statistics about test runs in a grid search log"""
# Created by Brendon Matusch, July 2018

import os
import sys

from utilities.verify_arguments import verify_arguments

# Verify that a path to the log file is passed
verify_arguments('log file path')
# Get it from the command line arguments
log_path = os.path.expanduser(sys.argv[1])
