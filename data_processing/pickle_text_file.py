#!/usr/bin/env python3
"""A script for converting data serialized as a text file to the Pickle format that is loaded when running training or analysis"""
# Created by Brendon Matusch, July 2018

import itertools
import os
import pickle
import shlex

from collections import namedtuple
from functools import reduce

from data_processing.bubble_data_point import BubbleDataPoint
from data_processing.event_data_set import EVENT_FILE_PATH

# A dictionary corresponding the last letter of a format string to the type of that element
TYPES_BY_FORMAT = {
    'd': int,
    'e': float,
    'f': float,
    's': str
}

# Open the text file and read it line by line
with open(os.path.expanduser('~/merged_all.txt')) as text_file:
    # Read and ignore the first line, which contains useless description
    text_file.readline()
    # Read the second line, strip it, and split it on whitespace; it includes the names and dimensions of attributes in the file
    attributes = text_file.readline().strip().split()
    # Iterate over the attributes, adding to lists of names and corresponding numbers of sub-elements
    attribute_names = []
    attribute_elements = []
    for attribute in attributes:
        # If there are no brackets containing numbers of elements, add the name and set the number of elements to 1
        if '(' not in attribute:
            attribute_names.append(attribute)
            attribute_elements.append(1)
        # Otherwise, we need to handle a multi-element attribute
        else:
            # Split the attribute description by the open bracket and extract the name, which is before the brackets
            attribute_split = attribute.split('(')
            attribute_names.append(attribute_split[0])
            # After the open bracket is a comma-separated list of numbers with a close bracket at the end; separate the dimension numbers
            dimensions = [int(number_string.strip()) for number_string in attribute_split[1][:-1].split(',')]
            # Multiply each of the dimensions together to get the total number of elements in this attribute, and add it to the list
            elements = reduce(lambda first, second: first * second, dimensions)
            attribute_elements.append(elements)
    # Read and split the next line, which contains format strings for every element in the lines ahead
    format_strings = text_file.readline().strip().split()
    # Take only the last letter of each format string, which denotes the type, and get the types corresponding to these from the dictionary
    data_types = [TYPES_BY_FORMAT[format_string[-1]] for format_string in format_strings]
    # Read and ignore the next 3 lines, which do not contain any data
    for _ in range(3):
        text_file.readline()
    # Create a list to add the bubbles to
    bubbles = []
    # Iterate over line indices, reading data, until breaking at the end of the file
    for line_index in itertools.count():
        # Notify the user every 100 lines
        if line_index % 100 == 0:
            print(f'Loaded {line_index} lines')
        # Read a line of data; stop iteration if it is empty, meaning we have reached the end of the file
        data_line = text_file.readline()
        if not data_line:
            break
        # Otherwise, strip and split it using the shlex module to keep strings in quotes intact
        data_strings = shlex.split(data_line.strip())
        # Create a generator to convert the strings to their corresponding data types
        data_points = (data_type(data_string) for data_type, data_string in zip(data_types, data_strings))
        # Iterate over the names and numbers of elements in the attributes, adding the data to an event dictionary
        event_dictionary = {}
        for name, num_elements in zip(attribute_names, attribute_elements):
            # If there is only one element, take it from the generator and add it to the dictionary; if there are multiple, take and add them as a list
            event_dictionary[name] = next(data_points) if num_elements == 1 else [next(data_points) for _ in range(num_elements)]
        # Create a named tuple object out of the dictionary
        event_object = namedtuple('Event', sorted(event_dictionary.keys()))(**event_dictionary)
        # Convert it to a bubble with the corresponding unique line index
        bubbles.append(BubbleDataPoint(event_object, line_index))
# Serialize the list to a Pickle binary file and notify the user
with open(EVENT_FILE_PATH, 'wb') as pickle_file:
    pickle.dump(bubbles, pickle_file)
print('Saved as', EVENT_FILE_PATH)
