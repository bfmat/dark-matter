#!/usr/bin/env python3
"""Create a human-readable PDF containing graphs from the experimental data logs and explanatory indices"""
# Created by Brendon Matusch, December 2018

import datetime
import functools

import matplotlib.pyplot as plt
from pylatex import Document, Section, Figure, NoEscape, Subsection

# Create a LaTeX document to add everything to
document = Document('manual_optimization_log', geometry_options={'margin': '1in'})

# LEARNING CURVE GRAPHING SECTION
# Add a title and description explaining this section
with document.create(Section('Learning Curves')):
    document.append('This section contains learning curves documenting the training of models tested during empirical experimentation.')
    # Skip to the next page to begin printing graphs
    document.append(NoEscape('\\newpage'))
# Load the training log index file
with open('../experimental_data/training_logs/index.txt') as file:
    training_log_index = file.readlines()
# Separate the log entries into log file names and descriptions
file_names, descriptions = zip(*[line.strip().split(': ', 1) for line in training_log_index])
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
for file_name, description in zip(file_names, descriptions):
    # Get the date from the beginning of the file name (separated by underscores)
    date = datetime.date(*[int(number) for number in file_name.split('_')[:3]])
    # If the date is new, update it and create a section header
    if date != last_date:
        last_date = date
        document.append(Section(str(date), numbering=False))
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
    # Create a figure and main axis to plot on
    figure, first_axis = plt.subplots()
    # Set the axis labels
    first_axis.set_xlabel('Epoch')
    first_axis.set_ylabel('Mean Squared Error Loss')
    # Create a list for the axes to plot on
    axes = [first_axis, None, None, first_axis, None, None]
    # If there are more data points than just loss, create a second Y axis
    if accuracy_present or abs_error_present:
        second_axis = first_axis.twinx()
        # Set its label (naming it according to the second series)
        if accuracy_present:
            second_axis.set_ylabel('Accuracy')
        else:
            second_axis.set_ylabel('Absolute Error')
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
    # If there is only one axis, create a legend
    if (not accuracy_present) and (not abs_error_present):
        first_axis.legend(loc='center right')
    # Otherwise, we have to make a combined legend
    else:
        # Add the lines together
        combined_lines = functools.reduce(lambda x, y: x + y, lines)
        # Get the labels from the combined lines
        labels = [line.get_label() for line in combined_lines]
        first_axis.legend(combined_lines, labels, loc='center right')
    # Get a name for this plot subsection, taking it from the file name after the date and before the .out file extension
    subsection_name = file_name.split('_', 3)[-1][:-4].replace('_', ' ').title()
    # Create a section in the document for this plot (using the file name)
    with document.create(Subsection(subsection_name, numbering=False)):
        # Add the plot to the document
        with document.create(Figure(position='h!')) as plot:
            # Make it the full width of the text
            plot.add_plot(width=NoEscape('\\textwidth'))
            # Add the description as a caption to the plot
            plot.add_caption(description)
        # Add a new page before the next plot
        document.append(NoEscape('\\newpage'))
# Generate a PDF containing all of the plots
document.generate_pdf()
