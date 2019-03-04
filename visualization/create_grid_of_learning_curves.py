#!/usr/bin/env python3
"""Create a large grid of learning curves to visually demonstrate the manual optimization process"""
# Created by Brendon Matusch, March 2019

import functools
import itertools

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# Width and height of the grid of graphs to generate
WIDTH = 14
HEIGHT = 10

# Create a grid of main axes to plot on, with a predefined size
figure, first_axes = plt.subplots(HEIGHT, WIDTH, 'none', 'none', figsize=(28, 20))

# Load the training log index file
with open('../experimental_data/training_logs/index.txt') as file:
    training_log_index = file.readlines()
# Separate the log entries into log file names and descriptions
file_names, descriptions = zip(*[line.strip().split(': ', 1) for line in training_log_index])
for n in file_names:
    print(n)
# Convert the resulting tuples to lists
file_names = list(file_names)
descriptions = list(descriptions)
# Create a list of indices of entries to delete based on their file name
delete_indices = [file_names.index(delete_file_name) for delete_file_name in [
    '2018_07_10_initial_high_resolution_frequency_domain.out',
    '2018_07_21_iterative_cluster_nucleation_full_resolution_without_wall_cuts.out',
    '2018_07_24_high_resolution_frequency_grid_search.out',
    '2018_07_27_iterative_cluster_nucleation_grid_search.out.gz',
    '2018_07_27_time_of_flight_localization.out',
    '2018_07_28_waveform_grid_search.out',
    '2018_07_29_image_grid_search.out',
    '2018_08_01_ap_disagreements.out',
    '2018_08_02_iterative_cluster_nucleation_best_hyperparameters_from_grid_search.out',
    '2018_08_02_ap_similarity_means_by_run.out',
    '2018_08_13_triplets_nucleation_grid_search.out.gz',
    '2018_08_13_triplets_nucleation_grid_search_saved_validation_sets.out.gz',
    '2018_08_14_triplets_nucleation_best_configuration_standard_deviations.out',
    '2018_10_14_ap_triplets_accuracy.out',
    '2018_10_26_deap_real_regression.out'
]]
# Delete the indices of the file names, going backwards so the indexes do not change as the deletions are done
for index in reversed(sorted(delete_indices)):
    del file_names[index]
    del descriptions[index]
# Create a date variable for comparison
last_date = None
# Iterate over the files, creating graphs with descriptions
for index, file_name, description in zip(itertools.count(), file_names, descriptions):
    # Prepend the file name with the path to the log folder
    file_path = f'../experimental_data/training_logs/{file_name}'
    # Open the file and load its full contents
    with open(file_path) as file:
        file_contents = file.read()
    # If there are multiple training runs, take only the first
    if 'Trainable params' in file_contents:
        file_contents = file_contents.split('Trainable params')[1]
    # Split it into individual lines
    file_contents = file_contents.split('\n')
    # Take only the lines at the end of the epoch
    epoch_end_lines = [line for line in file_contents if 'step' in line]
    # Check for the presence of the main performance statistics (other than loss, which is always there)
    accuracy_present = 'acc' in epoch_end_lines[0]
    abs_error_present = 'mean_absolute_error' in epoch_end_lines[0]
    validation_present = 'val' in epoch_end_lines[0]
    # Based on each existing combination of attributes, set the word indices for [train loss, train accuracy, train abs error, val loss, val accuracy, val abs error]
    if accuracy_present:
        if validation_present:
            word_indices = [7, 10, None, 13, 16, None]
        else:
            word_indices = [7, 10, None, None, None, None]
    elif abs_error_present:
        # Validation is always present for mean absolute error runs
        word_indices = [7, None, 10, 13, None, 16]
    else:
        word_indices = [7, None, None, 10, None, None]
    # Create lists of series labels and colors to go along with the element indices
    series_labels = ['Train loss', 'Train accuracy', 'Train absolute error', 'Validation loss', 'Validation accuracy', 'Validation absolute error']
    colors = ['b', 'r', 'r', 'g', 'y', 'y']
    # Calculate the position of this plot within the grid
    x_position = index % WIDTH
    y_position = index // WIDTH
    # If we have surpassed the end of the grid, stop here
    if y_position >= HEIGHT:
        break
    # Get the relevant first axis out of the array
    first_axis = first_axes[y_position, x_position]
    # Turn off all ticks so the grid is not cluttered
    first_axis.get_xaxis().set_visible(False)
    first_axis.get_yaxis().set_visible(False)
    # Create a list for the axes to plot on
    axes = [first_axis, None, None, first_axis, None, None]
    # If there are more data points than just loss, create a second Y axis
    if accuracy_present or abs_error_present:
        second_axis = first_axis.twinx()
        # Turn off the Y ticks for this as well
        second_axis.get_yaxis().set_visible(False)
        # Include the new second axis in the list of axes
        axes = [first_axis, second_axis, second_axis, first_axis, second_axis, second_axis]
    # Create a list of lines to add the plots to
    lines = []
    # Iterate over word indices alongside the axes, series labels, and colors
    for word_index, axis, series_label, color in zip(word_indices, axes, series_labels, colors):
        # If there is no word index for this series (it does not exist), skip to the next
        if word_index is None:
            continue
        # Take the data points for this series from the epoch ending lines
        data_series = [float(line.strip().split()[word_index]) for line in epoch_end_lines]
        # Plot it on the current axis using the provided series label and color, adding the result to the list of lines
        lines.append(axis.plot(data_series, label=series_label, color=color))
# Display the grid of plots
plt.show()
