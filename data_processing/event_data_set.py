"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import os
import pickle
import random
from typing import Callable, List, Optional, Set, Tuple, Generator

import numpy as np

from data_processing.bubble_data_point import BubbleDataPoint, RunType, TriggerCause, load_bubble_audio


# The path to the Pickle data file which contains processed attributes of bubble events
EVENT_FILE_PATH = os.path.expanduser('~/merged.pkl')

# The number of examples to include in the validation set
VALIDATION_EXAMPLES = 128


class EventDataSet:
    """A bubble event data set class that is loaded from Pickle data, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    @staticmethod
    def load_data_from_file() -> List[BubbleDataPoint]:
        """Load and return all bubbles from the Pickle file"""
        # Simply load the list straight from the binary file and return it
        with open(EVENT_FILE_PATH, 'rb') as pickle_file:
            data_list = pickle.load(pickle_file)
        return data_list

    def __init__(
        self,
        # Keep only a certain set of run types in the data set
        keep_run_types: Optional[Set[RunType]]
    ) -> None:
        """Initializer that takes parameters that determine which data is loaded; None for the set of run types represents no filtering"""
        # Load the data from the Pickle file on disk
        events = self.load_data_from_file()
        # If there are run types provided, filter out data points that are not in the set
        if keep_run_types is not None:
            events = [
                event for event in events
                if event.run_type in keep_run_types
            ]
        # Run a series of filters on it (these are the universal filters used for both training and validation)
        events = [
            event for event in events
            if self.passes_standard_cuts(event)
        ]
        # Run cuts required only for validation on a copy of the list
        events_passing_validation_cuts = [
            event for event in events
            if self.passes_validation_cuts(event)
        ]
        # Add 2 copies of the events passing validation cuts to the original list of events, so they are weighted 3 times as heavily
        for _ in range(2):
            events += events_passing_validation_cuts
        # Choose a specified number of random examples from the list with validation cuts applied
        self.validation_events = random.sample(
            events_passing_validation_cuts, VALIDATION_EXAMPLES)
        # Remove all of the validation events from the original list of events
        events = [
            event for event in events
            if event not in self.validation_events
        ]
        # Randomize the order of the remaining events and move them into a global list
        random.shuffle(events)
        self.training_events = events

    @classmethod
    def passes_standard_cuts(cls, event: BubbleDataPoint) -> bool:
        """Determines whether an event passes the basic quality cuts run on all data"""
        # Run a series of binary cuts; if any of them fail, return False
        return (
            # Always filter out the garbage data
            event.run_type != RunType.GARBAGE
            # Keep only events captured due to the camera trigger and not timeouts, manual triggers, or auto-relaunches
            and event.trigger_cause == TriggerCause.CAMERA_TRIGGER
            # Always exclude events with a very large negative acoustic parameter (this is completely invalid)
            and event.logarithmic_acoustic_parameter > -100
            # Keep only events containing around one bubble based on the image and pressure transducer
            and event.num_bubbles_image <= 1
            and event.num_bubbles_pressure >= 0.7 and event.num_bubbles_pressure <= 1.3
            # Do not use events within the first 25s after reaching target pressure
            and event.time_since_target_pressure > 25
        )

    @classmethod
    def passes_validation_cuts(cls, event: BubbleDataPoint) -> bool:
        """Determines whether an event passes the cuts necessary for validation, which optimize discrimination using the acoustic parameter"""
        # Accept only events that pass both the fiducial cuts and the audio-based wall cuts
        return cls.passes_fiducial_cuts(event) and cls.passes_audio_wall_cuts(event)

    @staticmethod
    def passes_fiducial_cuts(event: BubbleDataPoint) -> bool:
        """Determines whether an event passes the fiducial cuts which define an area of the vessel excluding the horizontal and vertical extremes"""
        # Omit events above a certain height (too near the surface of the vessel)
        return event.z_position <= 523 and (
            # Accept several different possible regions that the bubble can occupy around the center of the tank
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

    @staticmethod
    def passes_audio_wall_cuts(event: BubbleDataPoint) -> bool:
        """Determines whether an event passes the cuts that distinguish between bulk and wall events based on the pressure transducer and the piezo frequency distribution"""
        # Run a cut on the pressure transducer value not corrected for position, to remove more wall-like events
        return (
            event.pressure_not_position_corrected < 1.3
            and event.pressure_not_position_corrected > 0.7
        ) and (
            # Run a cut on the AP12 variable, which is based on the frequency distribution and distinguishes wall events
            event.acoustic_parameter_12 < 300
            and event.acoustic_parameter_12 > 45
        )

    @classmethod
    def load_specific_indices(cls, specific_unique_indices: List[int]) -> List[BubbleDataPoint]:
        """An alternative loading method that does not do any filtering or sorting (except for the standard cuts), but rather loads only events with specific defined indices"""
        # Load all events from the Pickle file on disk
        all_data = cls.load_data_from_file()
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
        # Return the list of filtered events
        return sorted_data

    def banded_frequency_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the banded frequency domain data, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Create flattened training arrays and binary ground truth arrays for both training and validation
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                # Flatten the banded frequency domain information (without positional corrections) into single-dimensional arrays, and stack all of the examples into an array
                # Take only the last two piezos, as the first one does not work
                np.stack([
                    np.concatenate([
                        event.banded_frequency_domain_raw[1:].flatten(),
                        [event.x_position, event.y_position, event.z_position]
                    ])
                    for event in events
                ]),
                # Normal background radiation data represents alpha particles in the ground truth array, and everything else represents neutrons
                np.array([event.run_type == RunType.LOW_BACKGROUND
                          for event in events])
            )
            for events in [self.training_events, self.validation_events]
        ]
        # Return both components of both datasets
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def audio_alpha_classification(self, loading_function: Callable[[BubbleDataPoint], List[np.ndarray]], include_positions: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the audio data from a provided loading function, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Iterate over the training and validation events, with corresponding lists to add audio and position inputs and ground truths to
        training_audio_inputs = []
        training_position_inputs = []
        training_ground_truths = []
        validation_audio_inputs = []
        validation_position_inputs = []
        validation_ground_truths = []
        for events, audio_inputs, position_inputs, ground_truths in zip(
            [self.training_events, self.validation_events],
            [training_audio_inputs, validation_audio_inputs],
            [training_position_inputs, validation_position_inputs],
            [training_ground_truths, validation_ground_truths]
        ):
            # Iterate over the events, loading audio and ground truth data
            for event in events:
                # Try to load the audio corresponding to this event
                audio = loading_function(event)
                # If an empty list is returned, continue to the next iteration
                if not audio:
                    continue
                # Otherwise, add the audio waveform to the list
                audio_inputs.append(audio[0])
                # Add the spatial position of the bubble to the list of inputs
                position_inputs.append([
                    event.x_position,
                    event.y_position,
                    event.z_position
                ])
                # Add a corresponding ground truth, True if this is from the alpha data set and false otherwise
                ground_truths.append(event.run_type == RunType.LOW_BACKGROUND)
        # Next, convert the lists into NumPy arrays and return them
        # Combine the audio inputs with the position inputs if the feature is enabled
        if include_positions:
            return (
                [
                    np.array(training_audio_inputs),
                    np.array(training_position_inputs)
                ],
                np.array(training_ground_truths),
                [
                    np.array(validation_audio_inputs),
                    np.array(validation_position_inputs)
                ],
                np.array(validation_ground_truths)
            )
        # Otherwise, just return the audio data
        else:
            return (
                np.array(training_audio_inputs),
                np.array(training_ground_truths),
                np.array(validation_audio_inputs),
                np.array(validation_ground_truths)
            )

    def arbitrary_alpha_classification_generator(
        self,
        data_converter: Callable[[BubbleDataPoint], List[np.ndarray]],
        storage_size: int,
        batch_size: int,
        examples_replaced_per_batch: int,
        custom_training_data: Optional[List[BubbleDataPoint]]=None,
        ground_truth: Optional[Callable[[BubbleDataPoint], bool]]=None
    ) -> Tuple[Callable[[], Generator[Tuple[np.ndarray, np.ndarray], None, None]], np.ndarray, np.ndarray]:
        """Return a generator which produces arbitrary training data for each bubble, with corresponding binary classification ground truths into neutrons and alpha particles; alongside it, return arrays of validation data"""

        # If None is provided for the ground truth calculation function, assume it is discrimination between low background and calibration runs
        if ground_truth is None:
            def ground_truth(bubble):
                return bubble.run_type == RunType.LOW_BACKGROUND

        # If there is a training list provided, ignore the validation bubbles and just set it directly
        if custom_training_data is not None:
            validation_inputs = []
            validation_ground_truths = []
            self.training_events = custom_training_data
        # Otherwise, create training and validation data
        else:
            # Convert the validation bubbles right away and include the position data, and also get corresponding binary values for ground truth
            validation_inputs = [
                np.stack([
                    data_converter(bubble, use_synthesis=False)[0]
                    for bubble in self.validation_events
                ]),
                np.stack([
                    [bubble.x_position, bubble.y_position, bubble.z_position]
                    for bubble in self.validation_events
                ])
            ]
            validation_ground_truths = np.array([
                bubble.run_type == RunType.LOW_BACKGROUND
                for bubble in self.validation_events
            ])

        def generate_data() -> Generator[Tuple[List[np.ndarray], np.ndarray], None, None]:
            """The generator returned from the function"""
            # Create lists to store a changing set of main inputs, positional inputs, and ground truths in, so that they don't have to be reloaded for every batch
            main_inputs = []
            positional_inputs = []
            ground_truths = []
            # Iterate forever, loading and returning training examples and ground truth values
            while True:
                # If there are fewer examples than expected in the list, load some more
                while len(main_inputs) < storage_size:
                    # Choose one of the bubbles randomly
                    bubble = random.choice(self.training_events)
                    # Get examples for this bubble and add it to the list
                    bubble_examples = data_converter(
                        bubble, use_synthesis=False)
                    main_inputs += bubble_examples
                    # Add the position of the bubble on all 3 axes to the list
                    positional_inputs.append([
                        bubble.x_position,
                        bubble.y_position,
                        bubble.z_position
                    ])
                    # Add an equivalent number of binary values to the ground truth list, saying whether these examples represent alpha particles or neutrons
                    ground_truths += [(bubble.run_type ==
                                       RunType.LOW_BACKGROUND)] * len(bubble_examples)
                # Choose some random indices from the length of the training input and ground truth lists to return as a batch
                batch_indices = random.sample(
                    range(len(main_inputs)), batch_size)
                # Select lists of inputs and ground truths with these indices and convert them to NumPy arrays
                batch_main_inputs = np.array(main_inputs)[batch_indices]
                batch_positional_inputs = \
                    np.array(positional_inputs)[batch_indices]
                batch_ground_truths = np.array(ground_truths)[batch_indices]
                # Yield all 3 components of the data
                yield [batch_main_inputs, batch_positional_inputs], batch_ground_truths
                # Remove some random indices from both lists (they will be added back on the next iteration)
                for index in reversed(sorted(random.sample((range(len(main_inputs))), examples_replaced_per_batch))):
                    main_inputs.pop(index)
                    ground_truths.pop(index)

        # Return the generator alongside the validation inputs and ground truths
        return generate_data, validation_inputs, validation_ground_truths
