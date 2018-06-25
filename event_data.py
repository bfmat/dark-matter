"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import datetime
import itertools
import math
from enum import Enum
from typing import Set, Tuple

import numpy as np
import ROOT

# The path to the main data file which contains processed attributes of bubble events
EVENT_FILE_PATH = "/opt/merged_all_all.root"

# A threshold for the logarithmic acoustic parameter, which approximately discriminates between neutrons (below) and alpha particles (above)
ACOUSTIC_PARAMETER_THRESHOLD = 1.5


class RunType(Enum):
    """An enumeration of the possible run types, including various distinct radiation sources; numbers correspond to the numeric representations in the data table"""
    LOW_BACKGROUND = 0  # Run with normal background radiation
    AMERICIUM_BERYLLIUM = 2  # Calibration with AmBe source at the end of the source tube
    ACOUSTICS_BLINDED = 10  # Run with acoustic information blinded
    CALIFORNIUM_40CM = 14  # Cf source at 40cm from the bottom of the source tube
    CALIFORNIUM_60CM = 15  # Cf source at 60cm from the bottom of the source tube
    BARIUM_100CM = 21  # Ba source at 100cm from the bottom of the source tube
    BARIUM_40CM = 22  # Ba source at 45cm from the bottom of the source tube
    GARBAGE = 99  # Invalid data or unknown run type


class TriggerCause(Enum):
    """An enumeration of the possible causes of the recording trigger"""
    CAMERA_TRIGGER = 0  # The normal optical trigger when bubbles are observed
    TIMEOUT = 2  # The maximum time permitted for an event was reached
    MANUAL_OR_RELAUNCH = 3  # Either a manual trigger, or an auto-relaunch due to a problem


class EventDataSet:
    """A bubble event data set class that is loaded from CERN ROOT data as well as audio recordings and images, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    def __init__(self,
                 # Keep only a certain set of run types in the data set
                 keep_run_types: Set[RunType],
                 # Remove events containing multiple bubbles
                 filter_multiple_bubbles: bool = True
                 ) -> None:
        """Initializer that takes parameters that determine which data is loaded"""
        # Open the event file and get the main tree
        # These cannot be in the same line or a segmentation fault will occur
        event_file = ROOT.TFile(EVENT_FILE_PATH)
        tree = event_file.Get('T')
        # Iterate over the tree and convert the events to a custom data class
        events_data = [EventDataPoint(event) for event in tree]
        # Run a series of filters on the data
        events_data = [
            event for event in events_data
            # Always filter out the garbage data
            if event.run_type != RunType.GARBAGE
            # Keep only events with the specified set of run types
            and event.run_type in keep_run_types
            # Keep only events captured due to the camera trigger and not timeouts, manual triggers, or auto-relaunches
            and event.trigger_cause == TriggerCause.CAMERA_TRIGGER
            # Keep only events within a certain vertical range
            and event.z_position >= 0
            and event.z_position <= 523
            # Exclude events in the low background runs with an acoustic parameter above a threshold
            and not (event.run_type == RunType.LOW_BACKGROUND and event.logarithmic_acoustic_parameter > ACOUSTIC_PARAMETER_THRESHOLD)
            # Exclude events in the calibration runs with an acoustic parameter below that same threshold
            and not (event.run_type != RunType.LOW_BACKGROUND and event.logarithmic_acoustic_parameter < ACOUSTIC_PARAMETER_THRESHOLD)
        ]
        # Keep only events containing one bubble if the filter is enabled
        if filter_multiple_bubbles:
            events_data = [
                event for event in events_data
                if event.num_bubbles == 1
            ]
        # Transfer the filtered list to a global variable so it can be converted to training data elsewhere
        self.events_data = events_data

    def banded_frequency_alpha_classification(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the banded frequency domain data, with corresponding binary classification ground truths into neutrons and alpha particles"""
        # Flatten the banded frequency domain information into single-dimensional arrays, and stack all of the examples into an array
        flat_banded_data = np.stack([
            event.banded_frequency_domain.flatten()
            for event in self.events_data
        ])
        # Normal background radiation data represents alpha particles in the ground truth array, and everything else represents neutrons
        alpha_ground_truths = np.array([
            event.run_type == RunType.LOW_BACKGROUND
            for event in self.events_data
        ])
        # Return the banded frequency domain data and the ground truths together
        return flat_banded_data, alpha_ground_truths


class EventDataPoint:
    """A bubble event data class that contains all of the data necessary for training, for a single bubble event"""

    def __init__(self, event: ROOT.TTree) -> None:
        """Initializer that takes a single event in the CERN ROOT format and extracts the relevant data"""
        # The run identifier is in the format YYYYMMDD_RR (R is the run within that day); parse it to get a date and a run number
        run_identifier = event.run
        year = int(run_identifier[:4])
        month = int(run_identifier[4:6])
        day = int(run_identifier[6:8])
        self.date = datetime.date(year=year, month=month, day=day)
        # At the end of the string, there are often strange characters; iterate up to the last one which is a numeric digit, and use that to cut out the run number
        run_number_end_index = 10
        for character_index in itertools.count(run_number_end_index + 1):
            # Subtract one from the index because indexing in Python is exclusive for the end
            if run_identifier[character_index - 1].isdigit():
                run_number_end_index = character_index
            else:
                break
        self.run_number = int(run_identifier[9:run_number_end_index])
        # Get the event number within the run, which starts at 0
        self.event_number = event.ev
        # Assign the run type based on the raw numeric value, assuming garbage if the value is not present in the enumeration
        raw_run_type = event.run_type
        self.run_type = RunType(raw_run_type) \
            if raw_run_type in set(possible_run_type.value for possible_run_type in RunType) \
            else RunType.GARBAGE
        # Likewise for the cause of the recording trigger, assuming a manual trigger or relaunch due to a problem
        raw_trigger_cause = event.trigger_main
        self.trigger_cause = TriggerCause(raw_trigger_cause) \
            if raw_trigger_cause in set(possible_trigger_cause.value for possible_trigger_cause in TriggerCause) \
            else TriggerCause.MANUAL_OR_RELAUNCH
        # Get the position-corrected banded frequency domain representation of the audio as an array of strength values
        # It has to be converted to a list first; NumPy reads the length incorrectly
        banded_array = np.array(list(event.piezo_E_PosCor))
        # Reshape it into the format (time bin, frequency bin, piezo channel) where there are 3 time bins, 8 frequency bins, and 3 piezo channels
        self.banded_frequency_domain = np.reshape(banded_array, (3, 8, 3))
        # Get the number of bubbles present in the bubble, calculated through image matching
        self.num_bubbles = event.nbub
        # Get the approximated position of the bubble in 3 dimensions
        self.x_position = event.X
        self.y_position = event.Y
        self.z_position = event.Z
        # Get the horizontal and depth-wise distances from the bubble to the wall
        self.horizontal_distance_to_wall = event.Dwall
        self.depth_wise_distance_to_wall = event.Dwall_horiz
        # Compute the logarithmic acoustic parameter, which is used to sort background events out of the calibration runs
        # Substitute a large negative number if the raw value is zero or negative
        self.logarithmic_acoustic_parameter = \
            math.log(event.acoustic_bubnum, 10) if event.acoustic_bubnum > 0 \
            else -10_000
