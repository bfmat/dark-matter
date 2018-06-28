"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import os
import pickle
import random
from typing import List, Optional, Set, Tuple

from data_processing.bubble_data_point import BubbleDataPoint, RunType, TriggerCause, load_bubble_images

import numpy as np

# The path to the Pickle data file which contains processed attributes of bubble events
EVENT_FILE_PATH = os.path.expanduser('~/merged.pkl')

# A threshold for the logarithmic acoustic parameter, which approximately discriminates between neutrons (below) and alpha particles (above)
ACOUSTIC_PARAMETER_THRESHOLD = 1.2

# The amount of data (out of 1) to remove for validation
VALIDATION_SPLIT = 0.2


class EventDataSet:
    """A bubble event data set class that is loaded from Pickle data as well as audio recordings and images, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    @staticmethod
    def load_data_from_file() -> List[BubbleDataPoint]:
        """Load and return all bubbles from the Pickle file"""
        # Simply load the list straight from the binary file and return it
        with open(EVENT_FILE_PATH, 'rb') as pickle_file:
            data_list = pickle.load(pickle_file)
        return data_list

    def __init__(self,
                 # Keep only a certain set of run types in the data set
                 keep_run_types: Optional[Set[RunType]],
                 # Remove events containing multiple bubbles
                 filter_multiple_bubbles: bool,
                 # Run cuts based on acoustic parameter
                 filter_acoustic_parameter: bool,
                 ) -> None:
        """Initializer that takes parameters that determine which data is loaded; None for the set of run types represents no filtering"""
        # Load the data and run a series of filters on it
        events_data = self.load_data_from_file()
        events_data = [
            event for event in events_data
            # Always filter out the garbage data
            if event.run_type != RunType.GARBAGE
            # Keep only events captured due to the camera trigger and not timeouts, manual triggers, or auto-relaunches
            and event.trigger_cause == TriggerCause.CAMERA_TRIGGER
            # Keep only events within a certain vertical range
            and event.z_position >= 0
            and event.z_position <= 523
            # Always exclude events with a very large negative acoustic parameter (this is completely invalid)
            and event.logarithmic_acoustic_parameter > -100
        ]
        # If there are run types provided, filter out data points that are not in the set
        if keep_run_types is not None:
            events_data = [
                event for event in events_data
                if event.run_type in keep_run_types
            ]
        # Run some acoustic parameter cuts if the filter is enabled
        if filter_acoustic_parameter:
            events_data = [
                event for event in events_data
                # Exclude events in the low background runs with an acoustic parameter above a threshold
                if not (event.run_type == RunType.LOW_BACKGROUND and event.logarithmic_acoustic_parameter > ACOUSTIC_PARAMETER_THRESHOLD)
                # Exclude events in the calibration runs with an acoustic parameter below that same threshold
                and not (event.run_type != RunType.LOW_BACKGROUND and event.logarithmic_acoustic_parameter < ACOUSTIC_PARAMETER_THRESHOLD)
                # Exclude all events with a significantly negative acoustic parameter
                and event.logarithmic_acoustic_parameter > 0.4
            ]
        # Keep only events containing one bubble if the filter is enabled
        if filter_multiple_bubbles:
            events_data = [
                event for event in events_data
                if event.num_bubbles == 1
            ]
        # Randomize the order of the events and divide them into global training and validation sets according to the predefined proportion
        random.shuffle(events_data)
        training_split = 1 - VALIDATION_SPLIT
        validation_start_index = int(len(events_data) * training_split)
        self.training_events = events_data[:validation_start_index]
        self.validation_events = events_data[validation_start_index:]

    @staticmethod
    def load_specific_indices(specific_unique_indices: List[int]):
        """An alternative loading method that does not do any filtering or sorting (except for the standard cuts), but rather loads only events with specific defined indices"""
        # Return type annotation is not possible because this class cannot be referenced inside itself
        # Create a new instance of this class with no filtering except for the cuts that are always done
        event_data_set = EventDataSet(
            keep_run_types=None,
            filter_multiple_bubbles=False,
            filter_acoustic_parameter=False
        )
        # Combine its training and validation data into one array
        all_data = event_data_set.training_events + event_data_set.validation_events
        # Filter out the bubbles whose unique indices are not in the provided list
        filtered_data = [
            bubble for bubble in all_data
            if bubble.unique_bubble_index in specific_unique_indices
        ]
        # Sort the list of bubbles so that its order is the same as that of the list of indices provided
        sorted_data = sorted(
            filtered_data,
            key=lambda bubble: specific_unique_indices.index(
                bubble.unique_bubble_index
            )
        )
        # Set the validation data list in the event data set with the filtered and sorted data (and empty the training data list), and return it
        event_data_set.training_events = []
        event_data_set.validation_events = sorted_data
        return event_data_set

    def banded_frequency_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the banded frequency domain data, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Create flattened training arrays and binary ground truth arrays for both training and validation
        (training_input, training_ground_truths), (validation_input, validation_ground_truths) = [
            (
                # Flatten the banded frequency domain information into single-dimensional arrays, and stack all of the examples into an array
                np.stack([event.banded_frequency_domain.flatten()
                          for event in events]),
                # Normal background radiation data represents alpha particles in the ground truth array, and everything else represents neutrons
                np.array([event.run_type == RunType.LOW_BACKGROUND
                          for event in events])
            )
            for events in [self.training_events, self.validation_events]
        ]
        # Return both components of both datasets
        return training_input, training_ground_truths, validation_input, validation_ground_truths

    def image_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return images for each bubble, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Create lists to add the images and ground truths to, for training and validation
        training_images = []
        training_ground_truths = []
        validation_images = []
        validation_ground_truths = []
        # Iterate over the training and validation bubble lists, and corresponding image and ground truth lists
        for bubbles, images, ground_truths in zip(
            [self.training_events, self.validation_events],
            [training_images, validation_images],
            [training_ground_truths, validation_ground_truths]
        ):
            # Iterate over the list of bubbles, converting them to images
            for bubble in bubbles:
                # Get the images of this bubble and add them to the list for training or validation, whichever we are currently on
                bubble_images = load_bubble_images(bubble)
                images += bubble_images
                # Add an equivalent number of binary values to the ground truth list, saying whether these images represent alpha particles or neutrons
                ground_truths.append(
                    [(bubble.run_type == RunType.LOW_BACKGROUND)]
                    * len(bubble_images)
                )
        # Return both components of both datasets, converted to NumPy arrays
        return np.array(training_images), np.array(training_ground_truths), np.array(validation_images), np.array(validation_ground_truths)
