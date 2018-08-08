"""A function for loading triplet events from Rn-222, Po-218, and Po-214 decays and classifying loud and quiet events"""
# Created by Brendon Matusch, August 2018

import datetime
from typing import Tuple

import numpy as np

from data_processing.bubble_data_point import BubbleDataPoint
from data_processing.event_data_set import EventDataSet

# A list of strings containing triplets from 2.4keV runs
_24_TRIPLETS = [
    'Triplet Rn-222 20170405_2/36 A: 5.30 | Po-218 20170405_2/38 A: 6.65 | Po-214 20170406_0/2 A: 17.22',
    'Triplet Rn-222 20170417_2/26 A: 5.73 | Po-218 20170417_2/27 A: 7.33 | Po-214 20170417_2/38 A: 16.43',
    'Triplet Rn-222 20170423_2/2 A: 5.10 | Po-218 20170423_2/3 A: 8.03 | Po-214 20170423_2/5 A: 12.20',
    'Triplet Rn-222 20170425_2/40 A: 5.63 | Po-218 20170425_2/42 A: 6.37 | Po-214 20170425_2/44 A: 12.43',
    'Triplet Rn-222 20170425_2/42 A: 6.37 | Po-218 20170425_2/44 A: 12.43 | Po-214 20170425_2/48 A: 5.11',
    'Triplet Rn-222 20170509_1/0 A: 6.02 | Po-218 20170509_1/1 A: 6.05 | Po-214 20170509_1/13 A: 21.88',
    'Triplet Rn-222 20170601_5/23 A: 7.51 | Po-218 20170601_5/27 A: 19.54 | Po-214 20170601_6/6 A: 4.72',
]

# A list of strings containing triplets from 3.3keV runs
_33_TRIPLETS = [
    'Triplet Rn-222 20161024_0/19 A: 6.01 | Po-218 20161024_0/20 A: 7.31 | Po-214 20161024_0/25 A: 25.39',
    'Triplet Rn-222 20161024_1/15 A: 7.38 | Po-218 20161024_1/17 A: 6.80 | Po-214 20161024_1/22 A: 8.34',
    'Triplet Rn-222 20161104_5/30 A: 5.01 | Po-218 20161104_5/31 A: 7.03 | Po-214 20161104_5/33 A: 20.97',
    'Triplet Rn-222 20161119_0/9 A: 3.99 | Po-218 20161119_0/10 A: 4.78 | Po-214 20161119_0/11 A: 7.81',
    'Triplet Rn-222 20161119_0/10 A: 4.78 | Po-218 20161119_0/11 A: 7.81 | Po-214 20161119_0/16 A: 4.12',
    'Triplet Rn-222 20161126_0/54 A: 5.25 | Po-218 20161126_0/55 A: 6.08 | Po-214 20161126_0/58 A: 16.51',
    'Triplet Rn-222 20161127_1/9 A: 6.57 | Po-218 20161127_1/10 A: 5.64 | Po-214 20161127_1/13 A: 19.61',
    'Triplet Rn-222 20161127_1/49 A: 6.52 | Po-218 20161127_1/50 A: 7.75 | Po-214 20161127_1/51 A: 9.42',
    'Triplet Rn-222 20161127_1/50 A: 7.75 | Po-218 20161127_1/51 A: 9.42 | Po-214 20161127_1/55 A: 8.48',
    'Triplet Rn-222 20161202_1/34 A: 4.51 | Po-218 20161202_1/35 A: 7.31 | Po-214 20161202_1/38 A: 5.70',
    'Triplet Rn-222 20161214_1/32 A: 5.63 | Po-218 20161214_1/33 A: 5.29 | Po-214 20161214_1/35 A: 20.80',
    'Triplet Rn-222 20161216_0/50 A: 6.23 | Po-218 20161216_0/51 A: 5.40 | Po-214 20161216_0/57 A: 16.25',
    'Triplet Rn-222 20161222_1/40 A: 5.07 | Po-218 20161222_1/41 A: 5.25 | Po-214 20161222_1/44 A: 19.27',
    'Triplet Rn-222 20161223_1/30 A: 4.94 | Po-218 20161223_1/32 A: 6.30 | Po-214 20161223_1/36 A: 17.60',
    'Triplet Rn-222 20161228_0/0 A: 4.91 | Po-218 20161228_0/1 A: 5.72 | Po-214 20161228_0/15 A: 7.16',
    'Triplet Rn-222 20161230_0/33 A: 4.32 | Po-218 20161230_0/34 A: 6.83 | Po-214 20161230_0/42 A: 6.67',
    'Triplet Rn-222 20170108_1/31 A: 6.11 | Po-218 20170108_1/33 A: 8.11 | Po-214 20170108_1/36 A: 5.79',
    'Triplet Rn-222 20170111_1/43 A: 6.23 | Po-218 20170111_1/44 A: 6.82 | Po-214 20170111_1/48 A: 22.85',
    'Triplet Rn-222 20170119_0/39 A: 4.09 | Po-218 20170119_0/40 A: 4.87 | Po-214 20170119_1/3 A: 17.07',
]

def load_triplet_classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load classification data for triplets as NumPy arrays"""
    # Combine the lists of triplets into one
    all_triplets = _24_TRIPLETS + _33_TRIPLETS
    # Create lists for (date, run number, event number) tuples of loud and quiet events, which are the classification categories
    loud_event_tuples = []
    quiet_event_tuples = []
    # Iterate over the triplets, loading them
    for triplet in all_triplets:
        # Split it by whitespace and take the 3 event strings out
        split_triplet = triplet.split()
        event_strings = [split_triplet[2], split_triplet[7], split_triplet[12]]
        # Iterate over the event strings, adding tuples to a list for this triplet
        triplet_events = []
        for event_string in event_strings:
            # Split the string into run identifier and event number
            run_identifier, event_number = event_string.split('/')
            # Convert the event number to an integer
            event_number = int(event_number)
            # The run identifier is in the format YYYYMMDD_RR (R is the run within that day); parse it to get a date and a run number
            year = int(run_identifier[:4])
            month = int(run_identifier[4:6])
            day = int(run_identifier[6:8])
            date = datetime.date(year=year, month=month, day=day)
            run_number = int(run_identifier[9:])
            # Add the three values together to the list for this triplet
            triplet_events.append((date, run_number, event_number))
        # Add the triplet events to their corresponding event lists
        loud_event_tuples.append(triplet_events[0])
        quiet_event_tuples.append(triplet_events[1])
        quiet_event_tuples.append(triplet_events[2])
    # Load all events from disk
    all_events = EventDataSet.load_data_from_file()
    # Take only events that are in the tuple lists
    loud_events = [event for event in all_events if (event.date, event.run_number, event.event_number) in loud_event_tuples]
    quiet_events = [event for event in all_events if (event.date, event.run_number, event.event_number) in quiet_event_tuples]
    print(len(quiet_events))
    print(len(quiet_event_tuples))

load_triplet_classification_data()
