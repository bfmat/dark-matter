"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import os
import pickle
import random
from typing import Callable, List, Optional, Set, Tuple, Generator

import numpy as np

from data_processing.bubble_data_point import BubbleDataPoint, RunType, TriggerCause


# The path to the Pickle data file which contains processed attributes of bubble events
EVENT_FILE_PATH = os.path.expanduser('~/merged.pkl')

# A threshold for the logarithmic acoustic parameter, which approximately discriminates between neutrons (below) and alpha particles (above)
ACOUSTIC_PARAMETER_THRESHOLD = 1.2

# The amount of data (out of 1) to remove for validation in the non-generator training functions
VALIDATION_SPLIT = 0.2

# The absolute number of examples to reserve for validation in the training generator
GENERATOR_VALIDATION_EXAMPLES = 128


class EventDataSet:
    """A bubble event data set class that is loaded from Pickle data, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

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
                 # Filter out a certain proportion of the remaining data set randomly
                 filter_proportion_randomly: float,
                 # Filter out all events within certain areas of the tank near the walls
                 use_fiducial_cuts: bool
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
            # Always exclude events with a very large negative acoustic parameter (this is completely invalid)
            and event.logarithmic_acoustic_parameter > -100
            # TEMPORARY HACK: Only include events within a certain vertical range
            and event.z_position >= 0
            and event.z_position <= 523
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
        # Keep only events containing around one bubble based on the image and pressure transducer if the filter is enabled
        if filter_multiple_bubbles:
            events_data = [
                event for event in events_data
                if event.num_bubbles_image == 1
                and event.num_bubbles_pressure >= 0.8 and event.num_bubbles_pressure <= 1.2
            ]
        # Filter out events near the wall of the tank if the cuts are enabled
        if use_fiducial_cuts:
            events_data = [
                event for event in events_data
                # Omit events above a certain height (too near the surface of the vessel)
                if event.z_position <= 523
                # Accept several different possible regions that the bubble can occupy around the center of the tank
                and (
                    # The bubble can be near the wall, but must be within a constrained vertical range above the center of the tank
                    (
                        event.distance_to_wall > 6
                        and event.z_position > 0
                        and event.z_position <= 400
                    )
                    # The bubble can be near the wall and below the center of the tank, but has to be within a certain radius of the center
                    or (
                        event.distance_to_wall > 6
                        and event.z_position <= 0
                        and event.distance_from_center <= 100
                    )
                    # The bubble must be distant from the wall and below center, but must be far from the center
                    or (
                        event.distance_to_wall > 13
                        and event.distance_from_center > 100
                        and event.z_position <= 0
                    )
                    # The bubble can be distant from the wall and in the upper part of the tank
                    or (
                        event.distance_to_wall > 13
                        and event.z_position > 400
                    )
                )
            ]
        # Randomize the order of the events and remove a certain proportion
        random.shuffle(events_data)
        number_to_remove = int(filter_proportion_randomly * len(events_data))
        events_data = events_data[number_to_remove:]
        # Divide the events into global training and validation sets according to the predefined proportion
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
            filter_acoustic_parameter=False,
            filter_proportion_randomly=0,
            use_fiducial_cuts=False
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

    def arbitrary_alpha_classification_generator(
        self,
        data_converter: Callable[[BubbleDataPoint], List[np.ndarray]],
        storage_size: int,
        batch_size: int,
        examples_replaced_per_batch: int
    ) -> Tuple[Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]], np.ndarray, np.ndarray]:
        """Return a generator which produces arbitrary training data for each bubble, with corresponding binary classification ground truths into neutrons and alpha particles; alongside it, return arrays of validation data"""
        # Combine the training and validation lists together
        bubbles = self.training_events + self.validation_events
        # Split it into training and validation, but with a smaller number for validation; this is a hack required because Keras's validation generator feature does not work as documented
        training_bubbles = bubbles[GENERATOR_VALIDATION_EXAMPLES:]
        validation_bubbles = bubbles[:GENERATOR_VALIDATION_EXAMPLES]
        # Convert the validation bubbles right away, and also get corresponding binary values for ground truth
        validation_inputs = np.stack([
            data_converter(bubble)[0]
            for bubble in validation_bubbles
        ])
        validation_ground_truths = np.array([
            bubble.run_type == RunType.LOW_BACKGROUND
            for bubble in validation_bubbles
        ])

        def generate_data() -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
            """The generator returned from the function"""
            # Create lists to store a changing set of training examples and ground truths in, so that they don't have to be reloaded for every batch
            training_examples = []
            ground_truths = []
            # Iterate forever, loading and returning training examples and ground truth values
            while True:
                # If there are fewer examples than expected in the list, load some more
                while len(training_examples) < storage_size:
                    # Choose one of the bubbles randomly
                    bubble = random.choice(training_bubbles)
                    # Get examples for this bubble and add it to the list
                    bubble_examples = data_converter(bubble)
                    training_examples += bubble_examples
                    # Add an equivalent number of binary values to the ground truth list, saying whether these examples represent alpha particles or neutrons
                    ground_truths += [(bubble.run_type ==
                                       RunType.LOW_BACKGROUND)] * len(bubble_examples)
                # Choose some random indices from the length of the training input and ground truth lists to return as a batch
                batch_indices = random.sample(
                    range(len(training_examples)), batch_size)
                # Select lists of training examples and ground truths with these indices and convert them to NumPy arrays
                batch_inputs = np.array(training_examples)[batch_indices]
                batch_ground_truths = np.array(ground_truths)[batch_indices]
                # Yield both components of the data
                yield batch_inputs, batch_ground_truths
                # Remove some random indices from both lists (they will be added back on the next iteration)
                for index in reversed(sorted(random.sample((range(len(training_examples))), examples_replaced_per_batch))):
                    training_examples.pop(index)
                    ground_truths.pop(index)

        # Return the generator alongside the validation inputs and ground truths
        return generate_data, validation_inputs, validation_ground_truths
