#!/usr/bin/env python3
"""Graph the overall performance of a series of general machine learning techniques"""
# Created by Brendon Matusch, November 2018

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Create lists containing the performance statistics as well as the names of the configurations
PERFORMANCE_VALUES = [85, 96.5, 97.7, 99.2]
CONFIGURATIONS = ['Original NN Analysis', 'New NN Analysis', 'Iterative Cluster Nucleation', 'Gravitational Differentiation']
# Color and legend the entries according to whether they are supervised or semi-supervised learning
COLORS = ['gray', 'gray', 'green', 'green']
l = ['s', 's', 'm', 'm']

# Draw a bar graph using the performance statistics; do not use the configurations right away because they would be sorted alphabetically
plt.bar(range(len(PERFORMANCE_VALUES)), PERFORMANCE_VALUES, color=COLORS)
# Now set the configuration names, with words angled so they fit
plt.xticks(range(len(CONFIGURATIONS)), CONFIGURATIONS, rotation=10)
# Use percentage labels for the Y axis
ax = plt.axes()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Start at 70% so the difference is more obvious
plt.ylim(70, 100)
# Draw a legend for the two learning classes
supervised_patch = mpatches.Patch(color='gray', label='Supervised Learning')
semi_supervised_patch = mpatches.Patch(color='green', label='Semi-Supervised Learning')
plt.legend(handles=[supervised_patch, semi_supervised_patch])
# Display the graph on screen
plt.show()
