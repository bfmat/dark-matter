#!/usr/bin/env python3
"""Graph the overall performance of a series of general machine learning techniques for the DEAP experiment"""
# Created by Brendon Matusch, November 2018

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Create lists containing the performance statistics as well as the names of the configurations
# PERFORMANCE_VALUES = [87, 58.7, 58.8, 54.4]
PERFORMANCE_VALUES = [97, 99.1, 100, 99.3]
CONFIGURATIONS = ['Conventional Analysis', 'Dense Neural Network', 'Map Projection CNN', 'Topological CNN']
# Color and legend the entries according to whether they are supervised or semi-supervised learning
COLORS = ['gray', 'green', 'green', 'green']

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
plt.ylabel('Recall')
# Draw a legend for the two learning classes
conventional_patch = mpatches.Patch(color='gray', label='Conventional Analysis')
machine_learning_patch = mpatches.Patch(color='green', label='Machine Learning')
plt.legend(handles=[conventional_patch, machine_learning_patch])
# Display the graph on screen
plt.show()
