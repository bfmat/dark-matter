#!/usr/bin/env python3
"""Graph the overall performance of a series of general machine learning techniques for the DEAP experiment"""
# Created by Brendon Matusch, November 2018

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Create lists containing the performance statistics (in the form (mean, 8th percentile, 92nd percentile))
ALPHA = [(0.996, None, None), (0.992453871216953, 0.9843091334894615, 1.0), (0.9986762886158047, 0.996774193548387, 1.0),
         (0.9985087719298246, 0.9921052631578948, 1.0), (0.9959030977982591, 0.9914285714285715, 1.0)]
WIMP = [(0.91, None, None), (0.7835294571076522, 0.6601297764960347, 0.8723531004603303), (0.7672341098533079, 0.74375, 0.79915611814346),
        (0.9301859015688803, 0.6060606060606061, 1.0), (0.75745328295003, 0.7313432835820896, 0.7927631578947368)]
# The names of the configurations corresponding to all these lists
CONFIGURATIONS = ['Conventional Analysis', 'Dense Neural Network', 'Map Projection CNN', 'Topological CNN', 'Map Projection CNN Final Test']

# Create a figure with a predefined size
_, ax = plt.subplots(figsize=(8, 6))
# Use percentage labels for the Y axis
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Draw a bar graph using the performance statistics; do not use the configurations right away because they would be sorted alphabetically
plt.bar(range(len(PERFORMANCE_VALUES)), PERFORMANCE_VALUES, color=COLORS)
# Now set the configuration names, with words angled so they fit
plt.xticks(range(len(CONFIGURATIONS)), CONFIGURATIONS, rotation=10)
# Start at 90% so the difference is more obvious; go slightly above 100 so there is space
plt.ylim(ymin=90, ymax=100.5)
# Label the Y axis to specify what the numbers mean
plt.ylabel('Recall (higher is better)')
# Draw a legend for the two learning classes
conventional_patch = mpatches.Patch(color='gray', label='Conventional Analysis')
machine_learning_patch = mpatches.Patch(color='green', label='Machine Learning')
plt.legend(handles=[conventional_patch, machine_learning_patch], loc='lower left')
# Display the graph on screen
plt.show()
