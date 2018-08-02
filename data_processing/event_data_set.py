"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import os
import pickle
import random
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from sklearn.externals import joblib

from data_processing.bubble_data_point import BubbleDataPoint, RunType, TriggerCause, load_bubble_audio


# The paths to the data files which contain processed attributes of bubble events, for each of the two PICO-60 runs
RUN_1_PATH = os.path.expanduser('~/run1merged.pkl')
RUN_2_PATH = os.path.expanduser('~/run2alldata.pkl')

# The number of examples to include in the validation set
VALIDATION_EXAMPLES = 128


class EventDataSet:
    """A bubble event data set class that is loaded from Pickle data, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    # Lists for the initial input and ground truth indices corresponding to the examples of the training and validation sets
    training_initial_input_indices = None
    validation_initial_input_indices = None

    # A cache for a list of events loaded from the file
    data_from_file_cache = None

    @classmethod
    def load_data_from_file(cls, use_run_1: bool = False) -> List[BubbleDataPoint]:
        """Load and return all bubbles from the Pickle file for either PICO-60 run 1 or 2"""
        # If the data has already been loaded, just return that
        if cls.data_from_file_cache is not None:
            return cls.data_from_file_cache
        # Select the path to the Pickle file for either run 1 or 2, depending on which is selected
        path = RUN_1_PATH if use_run_1 else RUN_2_PATH
        # Use Pickle for run 1, but scikit-learn joblib for run 2
        loader = pickle if use_run_1 else joblib
        # Load the list straight from the binary file
        with open(path, 'rb') as pickle_file:
            data_list = loader.load(pickle_file)
        # Cache the data in case it needs to be used again, before returning it
        cls.data_from_file_cache = data_list
        return data_list

    def __init__(
        self,
        # Keep only a certain set of run types in the data set
        keep_run_types: Optional[Set[RunType]],
        # Use wall cuts on training and validation data
        use_wall_cuts: bool,
        # Whether to use PICO-60 run 1 or 2
        use_run_1: bool = False,
        # Whether or not to use the same temperature and pressure cuts done in the original analysis
        use_temperature_and_pressure_cuts: bool = False
    ) -> None:
        """Initializer that takes parameters that determine which data is loaded; None for the set of run types represents no filtering"""
        # Load the data from the Pickle file on disk according to the run selected
        events = self.load_data_from_file(use_run_1)
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
        # If wall cuts are enabled, run them on all events
        if use_wall_cuts:
            events = [
                event for event in events
                if self.passes_fiducial_cuts(event)
                and self.passes_audio_wall_cuts(event)
            ]
        # If temperature and pressure cuts are enabled, run them on all events
        if use_temperature_and_pressure_cuts:
            events = [
                event for event in events
                if self.passes_temperature_and_pressure_cuts(event)
            ]
        # Choose a specified number of random examples from the list with validation cuts applied
        self.validation_events = random.sample(events, VALIDATION_EXAMPLES)
        # Remove all of the validation events from the original list of events
        events = [
            event for event in events
            if event not in self.validation_events
        ]
        # Randomize the order of the remaining events and move them into a global list
        random.shuffle(events)
        self.training_events = events

    @staticmethod
    def passes_standard_cuts(event: BubbleDataPoint) -> bool:
        """Determines whether an event passes the basic quality cuts run on all data"""
        # Run a series of binary cuts; if any of them fail, return False
        return (
            # Always filter out the garbage data
            event.run_type != RunType.GARBAGE
            # Keep only events captured due to the camera trigger and not timeouts, manual triggers, or auto-relaunches
            and event.trigger_cause == TriggerCause.CAMERA_TRIGGER
            # Always exclude events with a very large negative acoustic parameter (this is completely invalid), but skip this cut if the event does not have the acoustic parameter attribute
            and (
                not hasattr(event, 'logarithmic_acoustic_parameter')
                or event.logarithmic_acoustic_parameter > -100
            )
            # Keep only events containing around one bubble based on the image and pressure transducer
            and event.num_bubbles_image <= 1
            # Skip the pressure cut if the event does not have the attribute
            and (
                not hasattr(event, 'num_bubbles_pressure')
                or (event.num_bubbles_pressure >= 0.7 and event.num_bubbles_pressure <= 1.3)
            )
            # Do not use events within the first 25s after reaching target pressure
            and event.time_since_target_pressure > 25
            # Do not use events where the position values are exactly -100 (this is presumably an error code)
            and event.x_position != -100
        )

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

    @staticmethod
    def passes_temperature_and_pressure_cuts(event: BubbleDataPoint) -> bool:
        """Determines whether or not an event passes the cuts that make sure it is near 16.05 degrees Celsius and that pressure is within the expected range for that temperature"""
        return (
            # The pressure should be set to 21
            event.pressure_setting == 21
            # The actual observed pressure should be within 1 of the set pressure
            and np.abs(event.pressure_readings[0] - event.pressure_setting) < 1
            # The temperature on the thermometer that is actually used for analysis should be within half a degree of 16.05
            and np.abs(event.temperature_readings[2] - 16.05) < 1
        )

    @classmethod
    def load_specific_indices(cls, specific_unique_indices: List[int]) -> List[BubbleDataPoint]:
        """An alternative loading method that does not do any filtering or sorting (except for the standard cuts), but rather loads only events with specific defined indices"""
        # Load all events from the Pickle file on disk
        all_data = cls.load_data_from_file()
        # Find a bubble corresponding to every index, ensuring that there is a 1 to 1 correspondence (that is, duplicate indices are respected)
        return [
            next(bubble for bubble in all_data if bubble.unique_bubble_index == unique_index)
            for unique_index in specific_unique_indices
        ]

    def banded_frequency_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the banded frequency domain data, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Create flattened training arrays and binary ground truth arrays for both training and validation
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                # Flatten the banded frequency domain information (with positional corrections) into single-dimensional arrays, and stack all of the examples into an array
                # Take only the last two piezos, as the first one does not work in the raw data
                np.stack([
                    event.banded_frequency_domain[1:, :, 2].flatten()
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

    def ap_simulation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened training arrays and Acoustic Parameter values for ground truths"""
        # Create flattened training arrays and binary ground truth arrays for both training and validation
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                # Flatten the banded frequency domain information (without positional corrections) into single-dimensional arrays, and stack all of the examples into an array
                # Take only the last two piezos, as the first one does not work
                np.stack([
                    np.concatenate([
                        event.banded_frequency_domain_raw[1:, :, 2].flatten()
                    ])
                    for event in events
                ]),
                # Use the Acoustic Parameter of each event for ground truth data
                np.array([event.logarithmic_acoustic_parameter for event in events])
            )
            for events in [self.training_events, self.validation_events]
        ]
        # Return both components of both datasets
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def position_from_time_zero(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return zero time data arrays, alongside position ground truths that the network should be trained to predict"""
        # Create flattened training arrays and ground truth arrays for both training and validation
        (training_inputs, training_ground_truths), (validation_inputs, validation_ground_truths) = [
            (
                # Stack together all of the zero time data, for all piezos, even the ones that were disabled in earlier analyses
                np.stack([
                    # Subtract the first piezo signal time from all of them, so they are either 0 or positive, and at least one is 0
                    event.piezo_time_zero - np.min(event.piezo_time_zero)
                    for event in events
                ]),
                # Stack together the 3 position values
                np.stack([
                    [event.x_position, event.y_position, event.z_position]
                    for event in events
                ])
            )
            for events in [self.training_events, self.validation_events]
        ]
        # Return both components of both datasets
        return training_inputs, training_ground_truths, validation_inputs, validation_ground_truths

    def position_from_waveform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return waveform audio arrays, alongside position ground truths that the network should be trained to predict"""
        # Iterate over the training and validation events, with corresponding lists to add audio and position ground truths to
        training_audio_inputs = []
        training_ground_truths = []
        validation_audio_inputs = []
        validation_ground_truths = []
        for events, audio_inputs, ground_truths in zip(
            [self.training_events, self.validation_events],
            [training_audio_inputs, validation_audio_inputs],
            [training_ground_truths, validation_ground_truths]
        ):
            # Iterate over the events, loading audio and ground truth data
            for event in events:
                # Try to load the audio corresponding to this event
                audio = load_bubble_audio(event)
                # If an empty list is returned, continue to the next iteration
                if not audio:
                    continue
                # Otherwise, add the audio waveform to the list
                audio_inputs += audio
                # Add the spatial position of the bubble to the list of ground truths
                ground_truths.append([
                    event.x_position,
                    event.y_position,
                    event.z_position
                ])
        # Next, convert the lists into NumPy arrays and return them
        return (
            np.array(training_audio_inputs),
            np.array(training_ground_truths),
            np.array(validation_audio_inputs),
            np.array(validation_ground_truths)
        )

    def audio_alpha_classification(self, loading_function: Callable[[BubbleDataPoint], List[np.ndarray]], include_positions: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the audio data from a provided loading function, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Iterate over the training and validation events, with corresponding lists to add audio and position inputs and ground truths to, as well as lists to add initial input indices to
        training_audio_inputs = []
        training_position_inputs = []
        training_ground_truths = []
        self.training_initial_input_indices = []
        validation_audio_inputs = []
        validation_position_inputs = []
        validation_ground_truths = []
        self.validation_initial_input_indices = []
        for events, audio_inputs, position_inputs, ground_truths, initial_input_indices in zip(
            [self.training_events, self.validation_events],
            [training_audio_inputs, validation_audio_inputs],
            [training_position_inputs, validation_position_inputs],
            [training_ground_truths, validation_ground_truths],
            [self.training_initial_input_indices, self.validation_initial_input_indices]
        ):
            # Iterate over the events, loading audio and ground truth data
            for event in events:
                # Try to load the audio corresponding to this event
                audio = loading_function(event)
                # If an empty list is returned, continue to the next iteration
                if not audio:
                    continue
                # The current length of the input list corresponds to the initial index for the next event
                initial_input_indices.append(len(audio_inputs))
                # Otherwise, add the audio waveform to the list
                audio_inputs += audio
                # Add the spatial position of the bubble to the list of inputs
                # There should be just as many copies as there are main inputs
                for _ in range(len(audio)):
                    position_inputs.append([
                        event.x_position,
                        event.y_position,
                        event.z_position
                    ])
                # Add a corresponding ground truth, True if this is from the alpha data set and false otherwise
                # If there are multiple input arrays returned, add that number of ground truths
                ground_truths += [event.run_type == RunType.LOW_BACKGROUND] * len(audio)
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
