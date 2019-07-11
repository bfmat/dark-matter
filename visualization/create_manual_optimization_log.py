#!/usr/bin/env python3
"""Create a human-readable PDF containing graphs from the experimental data logs and explanatory indices"""
# Created by Brendon Matusch, December 2018

import datetime
import functools
import json
import os

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from pylatex import Document, Section, Figure, NoEscape, Subsection, Package

# Import with different names so there is no conflict
from data_processing.experiment_serialization import load_test as load_test

# Create a LaTeX document to add everything to
document = Document('manual_optimization_log', geometry_options={'margin': '1in'})
# Disable page numbers
document.packages.append(Package('nopageno'))

# LEARNING CURVE GRAPHING SECTION
# Add a title and description explaining this section
with document.create(Section('Learning Curves', numbering=False)):
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
# Iterate over the files, creating graphs with descriptions
for file_name, description in zip(file_names, descriptions):
    # Get the date from the beginning of the file name (separated by underscores)
    date = datetime.date(*[int(number) for number in file_name.split('_')[:3]])
    # Create a section header with the date
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

# PREDICTION DISTRIBUTION GRAPHING SECTION
# Add a title and description explaining this section
with document.create(Section('Prediction Distributions', numbering=False)):
    document.append('This section contains prediction distributions, which visualize the predictions made by empirically developed models on validation data.')
    # Skip to the next page to begin printing graphs
    document.append(NoEscape('\\newpage'))
# Load the prediction distribution index file
with open('../experimental_data/validation_results/index.txt') as file:
    validation_index = file.readlines()
# Separate the log entries into folder names and descriptions
folder_names, descriptions = zip(*[line.strip().split(': ', 1) for line in validation_index])
# Convert the resulting tuples to lists
folder_names = list(folder_names)
descriptions = list(descriptions)
# Create a list of indices of entries to delete based on their folder name
delete_indices = [folder_names.index(delete_folder_name) for delete_folder_name in [
    '2018_07_24_high_resolution_frequency_grid_search',
    '2018_07_27_iterative_cluster_nucleation_grid_search',
    '2018_07_28_waveform_grid_search',
    '2018_07_29_image_grid_search',
    '2018_08_13_triplets_nucleation_grid_search_saved_validation_sets',
    '2018_10_26_deap_real_regression'
]]
# Delete the indices of the folder names, going backwards so the indexes do not change as the deletions are done
for index in reversed(sorted(delete_indices)):
    del folder_names[index]
    del descriptions[index]
# Iterate over the folder, creating graphs with descriptions
for folder_name, description in zip(folder_names, descriptions):
    # Prepend the folder name with the path to the log folder
    folder_path = f'../experimental_data/validation_results/{folder_name}'
    # If it is in fact a folder (some are just a single file)
    if os.path.isdir(folder_path):
        # Get an alphanumerically sorted list of all files in the folder
        files = sorted(os.listdir(folder_path))
        # If this is a pulse count training run, and there is a validation folder, take the files in that folder
        if 'pulse_count_validation' in files:
            folder_path = os.path.join(folder_path, 'pulse_count_validation')
            files = sorted(os.listdir(folder_path))
        # Take the last file in the folder for rendering, composing the full path
        file_path = os.path.join(folder_path, files[-1])
    # If it is only a single file, take its path directly
    else:
        file_path = folder_path
    # Load the file's contents to check whether it is for PICO or DEAP
    with open(file_path) as file:
        file_contents = file.read()
    # There is a 'unique_bubble_index' field for PICO JSON files
    is_deap = not ('unique_bubble_index' in file_contents)
    # If it is a PICO JSON file, create a prediction disctribution graph with comparison to AP
    if not is_deap:
        # Try to load the data set from the file, ignoring the run type ground truths
        try:
            events, _, network_outputs = load_test(file_path)
        # If there are bubble indices not found, this uses old datasets and cannot be graphed at present
        except StopIteration:
            continue
        # Get the acoustic parameter and neural network score data from the events
        acoustic_parameters, original_neural_network_scores = zip(
            *((event.logarithmic_acoustic_parameter, event.original_neural_network_score)
                for event in events)
        )
        # Calculate actual neutron/alpha ground truths based on AP
        ground_truths = np.array(acoustic_parameters) > 0.25
        # Convert the binary ground truth values to colors (red and blue) for graphing
        point_colors = [
            'r' if ground_truth else 'b'
            for ground_truth in ground_truths
        ]
        # Set the size of the resulting graph (it should be standard across all such graphs)
        plt.figure(figsize=(8, 6))
        # Scatter plot the acoustic parameter on the X axis and the neural network's predictions on the Y axis
        plt.scatter(
            x=acoustic_parameters,
            y=network_outputs,
            c=point_colors
        )
        # Create patches that describe the 2 differently colored classes
        background_patch = Patch(color='r', label='Alpha Particles')
        calibration_patch = Patch(color='b', label='Neutrons')
        # Display them in a legend
        plt.legend(handles=[background_patch, calibration_patch])
        # Label the X and Y axes
        plt.xlabel('Logarithmic Acoustic Parameter')
        plt.ylabel('Neural Network Prediction')
        # Draw a vertical line to represent the AP decision boundary
        plt.axvline(0.25, color='black')
    # If this is a DEAP JSON file, graph a histogram showing its prediction density
    else:
        # Reimplementation of the DEAP load_test function without identifiers (which are not consistently present)
        # Load the contents of the JSON file from the provided path
        with open(os.path.expanduser(file_path)) as input_file:
            input_list = json.load(input_file)
        # Iterate over the list of dictionaries describing the events, adding information to lists
        ground_truths = []
        network_outputs = []
        for event_information in input_list:
            # Get the ground truth and network output, and add them each to the corresponding list
            ground_truths.append(event_information['ground_truth'])
            network_outputs.append(event_information['network_output'])
        # Separate the network's outputs based on the value of the corresponding ground truth
        network_outputs_false = [output for output, ground_truth in zip(network_outputs, ground_truths) if not ground_truth]
        network_outputs_true = [output for output, ground_truth in zip(network_outputs, ground_truths) if ground_truth]
        # Set the size of the resulting graph (it should be standard across all such graphs)
        plt.figure(figsize=(8, 6))
        # Plot the network's outputs by ground truth in a histogram, labeling the 2 classes
        plt.hist([network_outputs_false, network_outputs_true], bins=16, label=['Simulated WIMP Events', 'Simulated Neck Alpha Events'])
        # Label the axes of the graph
        plt.xlabel(r'Network Prediction (0 $\Rightarrow$ WIMP Events, 1 $\Rightarrow$ Neck Alpha Events)')
        plt.ylabel('Validation Event Count')
        # Enforce the X axis range from 0 to 1
        plt.xlim(0, 1)
        # Include a legend in the graph, explaining the colors
        plt.legend()
    # Get the date from the beginning of the folder name (separated by underscores)
    date = datetime.date(*[int(number) for number in folder_name.split('_')[:3]])
    # Create a section header with the date
    document.append(Section(str(date), numbering=False))
    # Get a name for this plot subsection, taking it from the folder name after the date
    subsection_name = folder_name.split('_', 3)[-1].replace('_', ' ').title()
    # If there is a .json file extension, remove it
    if folder_name.endswith('.json'):
        subsection_name = subsection_name[:-5]
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
