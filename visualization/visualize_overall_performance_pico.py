#!/usr/bin/env python3
"""Graph the overall performance of a series of general machine learning techniques for the PICO experiment"""
# Created by Brendon Matusch, November 2018

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# RAW DATA FOR ORIGINAL NN ANALYSIS
# Accuracy: [0.8046875, 0.8359375, 0.828125, 0.765625, 0.7578125, 0.8203125]
# Precision: [0.7549019607843137, 0.7835051546391752, 0.7962962962962963, 0.7, 0.7075471698113207, 0.7722772277227723]
# Recall: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# MSE (CWSD): [0.60794566699542, 0.5299834700383025, 0.6476162519709378, 0.5931055604078271, 0.608065520294811, 0.5913339464426839]

# Create lists containing the performance statistics (in the form (mean, 8th percentile, 92nd percentile))
ACCURACY = [(0.8020833333333334, 0.7578125, 0.8359375), (0.9713107638888889, 0.9295572916666667, 0.9921875), (0.921875, None, None),
            (0.9453125, None, None), (0.629746835443038, None, None), (0.9914062499999999, 0.98203125, 1.0), (0.9973958333333334, 0.9921875, 1.0), (0.9828645833333334, 0.9690625, 1.0)]
PRECISION = [(0.7524213015423129, 0.7, 0.7962962962962963), (0.9603736240440659, 0.8927487791089421, 1.0),
             (0.8876404494382022, None, None), (0.9213483146067416, None, None), (0.629746835443038, None, None), (0.990066119642678, 0.9826983589280337, 1.0), (0.996031746031746, 0.9880952380952381, 1.0), (0.9874765392592432, 0.9726728395061726, 1.0)]
RECALL = [(1.0, 1.0, 1.0), (0.9948717948717949, 0.9846153846153846, 1.0), (1.0, None, None), (1.0, None, None), (1.0, None, None),
          (0.99625, 0.9887500000000001, 1.0), (1.0, 1.0, 1.0), (0.9943567705611501, 0.986926406926407, 1.0)]
MSE = [(0.5963417360249971, 0.5299834700383025, 0.6476162519709378), (0.25819182151328907, 0.1360320276916941, 0.42540410048601757),
       (0.43786097319647915, None, None), (0.377014618633049, None, None), (1.0, None, None), (0.14178744505543836, 0.007247015186006504, 0.2710015019060907),
       (0.05361385226887008, 0.004770042351295833, 0.14950281253896586), (0.17623961468143692, 0.027541887095559853, 0.24165255370205946)]
# Zip all of these lists into separate high and low arrays
accuracy_mean, accuracy_low, accuracy_high = np.array(list(zip(*ACCURACY)))
precision_mean, precision_low, precision_high = np.array(list(zip(*PRECISION)))
recall_mean, recall_low, recall_high = np.array(list(zip(*RECALL)))
mse_mean, mse_low, mse_high = np.array(list(zip(*MSE)))
# Convert the high and low lists to errors, substituting 0 for None
for array, mean in [(accuracy_high, accuracy_mean), (accuracy_low, accuracy_mean), (precision_high, precision_mean), (precision_low, precision_mean), (recall_high, recall_mean), (recall_low, recall_mean), (mse_high, mse_mean), (mse_low, mse_mean)]:
    # Iterate over the indices of the array
    for index in range(len(array)):
        # If the value is None, set it to 0
        if array[index] is None:
            array[index] = 0
        # Otherwise, replace it with the absolute difference from the mean
        else:
            array[index] = np.abs(array[index] - mean[index])
# The names of the configurations corresponding to all these lists
CONFIGURATIONS = ['Original NN Analysis', 'Banded DFT', 'Full-Resolution DFT', 'Raw Waveform CNN', 'Image CNN',
                  'Iterative Cluster Nucleation', 'Gravitational Differentiation', 'Gravitational Differentiation Final Test']

# Get the locations for the bars
BAR_WIDTH = 0.2
accuracy_locations = np.arange(len(CONFIGURATIONS))
precision_locations = accuracy_locations + BAR_WIDTH
recall_locations = precision_locations + BAR_WIDTH
mse_locations = recall_locations + BAR_WIDTH
# Create a figure with a predefined size
_, ax = plt.subplots(figsize=(8, 6))
# Use percentage labels for the Y axis
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Draw a bar graph using the performance statistics; do not use the configurations right away because they would be sorted alphabetically
plt.bar(accuracy_locations, accuracy_mean * 100, width=BAR_WIDTH, yerr=(accuracy_low * 100, accuracy_high * 100), label='Accuracy')
plt.bar(precision_locations, precision_mean * 100, width=BAR_WIDTH, yerr=(precision_low * 100, precision_low * 100), label='Precision')
plt.bar(recall_locations, recall_mean * 100, width=BAR_WIDTH, yerr=(recall_low * 100, recall_high * 100), label='Recall')
# Now set the configuration names, with words angled so they fit
plt.xticks(precision_locations, CONFIGURATIONS, rotation=10)
# Start at 70% so the difference is more obvious
plt.ylim(60, 100)
# Label the Y axis to specify what the numbers mean
plt.ylabel('Accuracy/Precision/Recall')
# Twin the Y axis and plot the mean squared error
ax_mse = ax.twinx()
plt.bar(mse_locations, mse_mean, width=BAR_WIDTH, yerr=(mse_low, mse_high), label='Mean Squared Error Loss', color='C3')
# Label the secondary Y axis as well
plt.ylabel('Mean Squared Error Loss')
# Combine all of the lines from both Y axes together, and create a legend
main_lines, main_labels = ax.get_legend_handles_labels()
mse_lines, mse_labels = ax_mse.get_legend_handles_labels()
ax_mse.legend(main_lines + mse_lines, main_labels + mse_labels)
# Display the graph on screen
plt.show()
