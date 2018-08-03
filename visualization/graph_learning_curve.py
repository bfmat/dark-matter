#!/usr/bin/env python3
"""Load a training output file and graph the changing validation and training accuracy values over the course of the run"""
# Created by Brendon Matusch, August 2018

import os
import sys

import matplotlib.pyplot as plt

from utilities.verify_arguments import verify_arguments

# A path to a training output file and the title of the plot should be provided
verify_arguments('path to training output file', 'title of plot')
# Get the full path and load all lines in the file, stripping whitespace off both ends
with open(os.path.expanduser(sys.argv[1])) as file:
    lines = [line.strip() for line in file.readlines()]
# Take only the last line for each epoch, which contains the time per step
last_lines = [line for line in lines if 'step' in line]
# For each of those last lines, extract the training accuracy, which is after the identifier 'acc:' preceded and followed by whitespace
space_separated_words = [line.split() for line in last_lines]
training_accuracy_values = [float(line_words[line_words.index('acc:') + 1]) for line_words in space_separated_words]
# Check if there are validation accuracy values in the lines
if 'val_acc:' in space_separated_words[0]:
    # If so, load them in the same way
    validation_accuracy_values = [float(line_words[line_words.index('val_acc:') + 1]) for line_words in space_separated_words]
# Otherwise, they must be located in their own lines
else:
    # Only take the lines that only contain the validation accuracy, and split them into words
    validation_accuracy_line_words = [line.split() for line in lines if 'Validation accuracy:' in line]
    # The third and last word in the lines contains the validation accuracy
    validation_accuracy_values = [float(line_words[2]) for line_words in validation_accuracy_line_words]

# Plot the training accuracy values in yellow and validation in green; label them accordingly
plt.plot(training_accuracy_values, 'y', label='Training accuracy')
plt.plot(validation_accuracy_values, 'g', label='Validation accuracy')
# Draw a legend with the labels that have already been set
plt.legend()
# Set the title provided as an argument
plt.title(sys.argv[2])
# Label the accuracy and epoch axes
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# Display the graph on screen
plt.show()
