"""Code related to the custom bubble data format used instead of ROOT"""
# Created by Brendon Matusch, June 2018

from enum import IntEnum
import datetime
import itertools
import math
import os
import random
from typing import List

import numpy as np
from scipy.ndimage import imread

# The path in which all of the raw images and audio data are stored
RAW_DATA_PATH = os.path.expanduser('~/projects/rrg-kenclark/pico/30l-16-data')

# The side length in pixels of the square windows to crop out of the bubble chamber images
WINDOW_SIDE_LENGTH = 50

# The number of images to load out of the 21 possibilities for each bubble
IMAGES_PER_BUBBLE = 1


class RunType(IntEnum):
    """An enumeration of the possible run types, including various distinct radiation sources; numbers correspond to the numeric representations in the data table"""
    LOW_BACKGROUND = 0  # Run with normal background radiation
    AMERICIUM_BERYLLIUM = 2  # Calibration with AmBe source at the end of the source tube
    ACOUSTICS_BLINDED = 10  # Run with acoustic information blinded
    CALIFORNIUM_40CM = 14  # Cf source at 40cm from the bottom of the source tube
    CALIFORNIUM_60CM = 15  # Cf source at 60cm from the bottom of the source tube
    BARIUM_100CM = 21  # Ba source at 100cm from the bottom of the source tube
    BARIUM_40CM = 22  # Ba source at 45cm from the bottom of the source tube
    GARBAGE = 99  # Invalid data or unknown run type


class TriggerCause(IntEnum):
    """An enumeration of the possible causes of the recording trigger"""
    CAMERA_TRIGGER = 0  # The normal optical trigger when bubbles are observed
    TIMEOUT = 2  # The maximum time permitted for an event was reached
    MANUAL_OR_RELAUNCH = 3  # Either a manual trigger, or an auto-relaunch due to a problem


class BubbleDataPoint:
    """A bubble event data class that contains all of the data necessary for training, for a single bubble; there are multiple of these recorded for a multi-bubble event"""

    def __init__(self, root_event, unique_bubble_index: int) -> None:
        """Initializer that takes a single event in the CERN ROOT format and extracts the relevant data; it also takes a unique index for this bubble"""
        # Set a global variable with the provided unique index
        self.unique_bubble_index = unique_bubble_index
        # Get the timestamp that this event was recorded at
        self.timestamp = root_event.timestamp
        # The run identifier is in the format YYYYMMDD_RR (R is the run within that day); parse it to get a date and a run number
        run_identifier = root_event.run
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
        self.event_number = root_event.ev
        # Assign the run type based on the raw numeric value, assuming garbage if the value is not present in the enumeration
        raw_run_type = root_event.run_type
        self.run_type = RunType(raw_run_type) \
            if raw_run_type in set(possible_run_type.value for possible_run_type in RunType) \
            else RunType.GARBAGE
        # Likewise for the cause of the recording trigger, assuming a manual trigger or relaunch due to a problem
        raw_trigger_cause = root_event.trigger_main
        self.trigger_cause = TriggerCause(raw_trigger_cause) \
            if raw_trigger_cause in set(possible_trigger_cause.value for possible_trigger_cause in TriggerCause) \
            else TriggerCause.MANUAL_OR_RELAUNCH
        # Get the position-corrected banded frequency domain representation of the audio as an array of strength values
        # It has to be converted to a list first; NumPy reads the length incorrectly
        banded_array = np.array(list(root_event.piezo_E_PosCor))
        # Reshape it into the format (time bin, frequency bin, piezo channel) where there are 3 time bins, 8 frequency bins, and 3 piezo channels
        self.banded_frequency_domain = np.reshape(banded_array, (3, 8, 3))
        # Get the number of bubbles present in the event, calculated through image matching
        self.num_bubbles = root_event.nbub
        # Get the approximated position of the bubble in 3 dimensions
        self.x_position = root_event.X
        self.y_position = root_event.Y
        self.z_position = root_event.Z
        # Get the horizontal and depth-wise distances from the bubble to the wall
        self.horizontal_distance_to_wall = root_event.Dwall
        self.depth_wise_distance_to_wall = root_event.Dwall_horiz
        # Compute the logarithmic acoustic parameter, which is used to sort background events out of the calibration runs
        # Substitute a large negative number if the raw value is zero or negative
        self.logarithmic_acoustic_parameter = math.log(root_event.acoustic_bubnum, 10) if root_event.acoustic_bubnum > 0 \
            else -10_000
        # Get the X and Y positions of the bubble on each of the four cameras
        self.camera_positions = [
            (root_event.hori0, root_event.vert0),
            (root_event.hori1, root_event.vert1),
            (root_event.hori2, root_event.vert2),
            (root_event.hori3, root_event.vert3)
        ]


def load_bubble_images(bubble: BubbleDataPoint) -> List[np.ndarray]:
    """Given a bubble data point, load, crop, and return a list of windows which contain that bubble"""
    # Format the date and run number to get the folder name, padding one-digit month and day numbers with zeroes
    date_run_folder_name = f'{bubble.date.year}{bubble.date.month:0>2d}{bubble.date.day:0>2d}_{bubble.run_number}'
    # Paths to the image folders are inconsistent; sometimes they contain another folder named with the date inside the first folder, and sometimes they do not
    # Check if there is a folder inside the first one and format the path accordingly
    date_run_folder_path = os.path.join(RAW_DATA_PATH, date_run_folder_name)
    name_duplicated_path = os.path.join(
        date_run_folder_path, date_run_folder_name)
    if os.path.isdir(name_duplicated_path):
        date_run_folder_path = name_duplicated_path
    # Combine this with the event number to get the full path to the image folder
    image_folder_path = os.path.join(
        date_run_folder_path,
        str(bubble.event_number),
        'Images'
    )
    # Create a list to hold the images of this bubble
    bubble_images = []
    # Iterate over the number of each of the four cameras and the position of the bubble in this camera
    for camera_number, (bubble_x, bubble_y) in enumerate(bubble.camera_positions):
        # If either axis is less than zero, the bubble could not be found; skip to the next iteration
        if bubble_x < 0 or bubble_y < 0:
            continue
        # Otherwise, iterate over randomly chosen samples from the image numbers 50 to 70 inclusive, which are recorded after the bubble is detected
        for image_number in random.sample(range(50, 71), IMAGES_PER_BUBBLE):
            # Format the full path of this image
            image_path = os.path.join(
                image_folder_path,
                f'cam{camera_number}_image{image_number}.png'
            )
            # Attempt to load it as a NumPy array, skipping to the next iteration if it we are not permitted to load the image or if it does not exist
            try:
                full_image = imread(image_path)
            except (PermissionError, FileNotFoundError):
                continue
            # Round the bubble X and Y positions to integers so they can be used to index the image
            bubble_x_integer, bubble_y_integer = (
                int(round(position)) for position in [bubble_x, bubble_y]
            )
            # Crop a square out, centered at the integer position
            window = full_image[(bubble_x_integer-WINDOW_SIDE_LENGTH):(bubble_y_integer+WINDOW_SIDE_LENGTH),
                                (bubble_y_integer-WINDOW_SIDE_LENGTH):(bubble_y_integer+WINDOW_SIDE_LENGTH)]
            # Add the cropped window to the list of images
            bubble_images.append(window)
    # Return the list of cropped images
    return bubble_images
