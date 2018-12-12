#!/usr/bin/env python3
"""Create a human-readable PDF containing graphs from the experimental data logs and explanatory indices"""
# Created by Brendon Matusch, December 2018

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
# Prepend each of the file names with the path to the log folder
file_paths = [f'../experimental_data/training_logs/{file_name}' for file_name in file_names]
# Iterate over the files, creating graphs with descriptions
for file_path in file_paths:
    # Open the file and load its full contents
    with open(file_path) as file:
        file_contents = file.readlines()
    # Take only the lines at the end of the epoch
    epoch_end_lines = [line for line in file_contents if 'step' in line]
