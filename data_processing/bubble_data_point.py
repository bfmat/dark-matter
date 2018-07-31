"""Code related to the custom bubble data format used instead of ROOT"""
# Created by Brendon Matusch, June 2018

import datetime
import itertools
import math
import os
import sys

from enum import Enum, IntEnum, auto
from typing import List, Optional

import numpy as np
from skimage.io import imread

from data_processing.audio_domain_processing import time_to_frequency_domain

# A constant that defines which run we are currently using
USE_RUN_1 = False

# The path in which all of the raw images and audio data are stored
RAW_DATA_PATH = os.path.expanduser('~/projects/rrg-kenclark/pico/30l-16-data')

# The side length in pixels of the square windows to crop out of the bubble chamber images
WINDOW_SIDE_LENGTH = 50
# The start (inclusive) and end (exclusive) indices of the range of images to load and stack
START_IMAGE_INDEX = 45
END_IMAGE_INDEX = 55

# The number of piezo channels present in the audio files (not all of them work)
CHANNELS = 8

# The separations between the bands (in Hz) to split the frequency domain representation of each time band into
FREQUENCY_BANDS = [
    0,
    1.5e3,
    3.5e3,
    5e3,
    7.5e3,
    9e3,
    1.25e4,
    1.6e4,
    2e4,
    2.75e4,
    3.5e4,
    4.5e4,
    6e4,
    8e4,
    1e5,
    1.25e5,
    1.5e5,
    1.6e5,
    1.75e5,
    1.85e5,
    2.0e5
]

# The number of audio samples per second in raw recordings
SAMPLES_PER_SECOND = 400_000

# The shape of the piezo_E banded Fourier transform array
BANDED_FREQUENCY_DOMAIN_SHAPE = (9, 45, 3) if USE_RUN_1 else (3, 8, 3)


class RunType(Enum):
    """An enumeration of the possible run types, including various distinct radiation sources; numbers correspond to the numeric representations in the data table"""
    # Run with normal background radiation
    LOW_BACKGROUND = auto()
    # Calibration with AmBe neutron source at the end of the source tube
    AMERICIUM_BERYLLIUM = auto()
    # Californium neutron source at various distances from the bottom of the source tube
    CALIFORNIUM = auto()
    # Barium gamma source at various distances from the bottom of the source tube
    BARIUM = auto()
    # Cobalt gamma source at 45cm from the bottom of the source tube
    COBALT = auto()
    # Engineering, test and commissioning data which should not be used
    GARBAGE = auto()


# A dictionary mapping numeric run types to values in the enum
RUN_TYPE_DICTIONARY = {
    # Unblind background radiation data
    0: RunType.LOW_BACKGROUND,
    # Data with acoustics originally blinded, with a 200Hz camera frame rate
    10: RunType.LOW_BACKGROUND,
    # More data with acoustics originally blinded, with a 340Hz camera frame rate
    100: RunType.LOW_BACKGROUND,
    # Neutron calibration data with AmBe source at the bottom of the source tube
    2: RunType.AMERICIUM_BERYLLIUM,
    # Neutron calibration data with 252Cf source 40cm from the bottom of the source tube
    14: RunType.CALIFORNIUM,
    # Neutron calibration data with 252Cf source 60cm from the bottom of the source tube
    15: RunType.CALIFORNIUM,
    # Neutron calibration data with 252Cf source 80cm from the bottom of the source tube
    16: RunType.CALIFORNIUM,
    # Gamma calibration data with 133Ba source 100cm from the bottom of the source tube
    21: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 45cm from the bottom of the source tube
    22: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 120cm from the bottom of the source tube
    23: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 10cm from the bottom of the source tube
    24: RunType.BARIUM,
    # Gamma calibration data with 133Ba source 45cm from the bottom of the source tube, after the frame rate change
    32: RunType.BARIUM,
    # Gamma calibration data with 60Co source 45cm from the bottom of the source tube, after the frame rate change
    41: RunType.COBALT,
    # Engineering, test and commissioning data which should not be used
    99: RunType.GARBAGE
}


class TriggerCause(IntEnum):
    """An enumeration of the possible causes of the recording trigger"""
    CAMERA_TRIGGER = 0  # The normal optical trigger when bubbles are observed
    TIMEOUT = 2  # The maximum time permitted for an event was reached
    MANUAL_OR_RELAUNCH = 3  # Either a manual trigger, or an auto-relaunch due to a problem


class BubbleDataPoint:
    """A bubble event data class that contains all of the data necessary for training, for a single bubble; there are multiple of these recorded for a multi-bubble event"""

    def __init__(self, root_event, unique_bubble_index: int) -> None:
        """Initializer that takes a single event in the CERN ROOT format and extracts the relevant data; it also takes a unique index for this bubble, and it loads different attributes depending on whether it is PICO-60 run 1 or run 2"""
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
            if len(run_identifier) >= character_index and run_identifier[character_index - 1].isdigit():
                run_number_end_index = character_index
            else:
                break
        self.run_number = int(run_identifier[9:run_number_end_index])
        # Get the event number within the run, which starts at 0
        self.event_number = root_event.ev
        # Assign the run type by looking up the raw numeric value in the dictionary, assuming garbage if the value is not present in the dictionary
        if root_event.run_type in RUN_TYPE_DICTIONARY:
            self.run_type = RUN_TYPE_DICTIONARY[root_event.run_type]
        else:
            self.run_type = RunType.GARBAGE
        # Likewise for the cause of the recording trigger, assuming a manual trigger or relaunch due to a problem
        raw_trigger_cause = root_event.trigger_main
        self.trigger_cause = TriggerCause(raw_trigger_cause) \
            if raw_trigger_cause in set(possible_trigger_cause.value for possible_trigger_cause in TriggerCause) \
            else TriggerCause.MANUAL_OR_RELAUNCH
        # Do the same with the banded frequency domain representation without any position corrections
        banded_array = np.array(list(root_event.piezo_E))
        self.banded_frequency_domain_raw = np.reshape(banded_array, BANDED_FREQUENCY_DOMAIN_SHAPE)
        # Calculate the AP12 variable, which is used to detect wall events with greater accuracy
        self.acoustic_parameter_12 = np.mean(
            self.banded_frequency_domain_raw
            [[0, 2], :, :]
            [:, [0, 1], :]
            [:, :, 2]
        )
        # Get the approximated position of the bubble in 3 dimensions
        self.x_position = root_event.X
        self.y_position = root_event.Y
        self.z_position = root_event.Z
        # Get the intended numeric pressure setting
        self.pressure_setting = int(root_event.pset)
        # Get the temperature and pressure readings
        self.temperature_readings = list(root_event.ts)
        self.pressure_readings = list(root_event.pts)
        # Get the distance from the center of the jar from the perspective of a camera, replacing it with a very large value if it is negative (could not be found)
        if root_event.R2 >= 0:
            self.distance_from_center = math.sqrt(root_event.R2)
        else:
            self.distance_from_center = 100000
        # Get the minimum of the distances from the bubble to the wall on two axes
        self.distance_to_wall = min(root_event.Dwall, root_event.Dwall_horiz)
        # Get the number of bubbles present in the event, calculated through image matching
        self.num_bubbles_image = root_event.nbub
        # Get the time since the target pressure was reached
        self.time_since_target_pressure = root_event.te
        # There are certain attributes that are only loaded for run 2
        if not USE_RUN_1:
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
            # Get the approximated number of bubbles based on the pressure transducer
            self.num_bubbles_pressure = root_event.dytranCZ
            # Get the pressure transducer value not corrected for position
            self.pressure_not_position_corrected = root_event.dytranC
            # Get the neural network score predicted in the original PICO-60 paper
            self.original_neural_network_score = root_event.NN_score
            # Get the position-corrected banded frequency domain representation of the audio as an array of strength values
            # It has to be converted to a list first; NumPy reads the length incorrectly
            banded_array = np.array(list(root_event.piezo_E_PosCor))
            # Reshape it into the format (time bin, frequency bin, piezo channel) where there are 3 time bins, 8 frequency bins, and 3 piezo channels
            self.banded_frequency_domain = np.reshape(banded_array, BANDED_FREQUENCY_DOMAIN_SHAPE)
        # There are also some that are only loaded for run 1
        else:
            # Get the time that each of the piezos received the signal
            self.piezo_time_zero = root_event.piezo_t0


def bubble_data_path(bubble: BubbleDataPoint) -> str:
    """Given a bubble data point, return the path to the folder containing all of the data corresponding to that bubble"""
    # Format the date and run number to get the folder name, padding one-digit month and day numbers with zeroes
    date_run_folder_name = f'{bubble.date.year}{bubble.date.month:0>2d}{bubble.date.day:0>2d}_{bubble.run_number}'
    # Paths to the image folders are inconsistent; sometimes they contain another folder named with the date inside the first folder, and sometimes they do not
    # Check if there is a folder inside the first one and format the path accordingly
    date_run_folder_path = os.path.join(RAW_DATA_PATH, date_run_folder_name)
    name_duplicated_path = os.path.join(
        date_run_folder_path, date_run_folder_name)
    if os.path.isdir(name_duplicated_path):
        date_run_folder_path = name_duplicated_path
    # Combine this with the event number to get the full path to the data folder
    return os.path.join(
        date_run_folder_path,
        str(bubble.event_number)
    )


def load_bubble_audio(bubble: Optional[BubbleDataPoint], audio_file_path: Optional[str] = None) -> List[np.ndarray]:
    """Load an audio file in the raw binary format present in the PICO-60 data set, returning an array with dimensions the number of samples containing the data from only the working microphones by 2"""
    # If the audio waveform has already been loaded, and no specific path has been selected, return it
    if audio_file_path is None and hasattr(bubble, 'waveform'):
        return bubble.waveform
    # If a path is not provided, get it from the bubble
    if audio_file_path is None:
        # Get the path to the bubble data folder, and load the audio binary file within it
        audio_file_path = os.path.join(
            bubble_data_path(bubble),
            'fastDAQ_0.bin'
        )
    # Try to open the file for binary reading
    try:
        with open(audio_file_path, 'rb') as audio_file:
            # Read and ignore 4 bytes which comprise the header
            audio_file.read(4)
            # The next 2 bytes are the length of the string describing the channels
            channels_string_length = int.from_bytes(
                audio_file.read(2), sys.byteorder)
            # Read the channels description string from the file now, and decode it as a string
            channels_string = audio_file.read(channels_string_length).decode()
            # The channels string is a semicolon-separated list of all the channels of the file
            # An example: Piezo3;int16;1;Piezo4;int16;1;Piezo9;int16;1;Piezo7;int16;1;unused1;int16;1;FrameClock;int16;1;Cam0Trig;int16;1;Cam1Trig;int16;1;
            # First, split it into the semicolon-separated components
            components = channels_string.split(';')
            # Try to get the indices of piezos 3 and 7 (the only ones that work), integer dividing them by 3 because of the data type annotations (int16) and mysterious 1s
            try:
                piezo_3_index = components.index('Piezo3') // 3
                piezo_7_index = components.index('Piezo7') // 3
            # If one or both of the piezos are not present in the file, print an error message and return nothing
            except ValueError:
                print(f'File {audio_file_path} is missing piezo 3 and/or 7')
                return []
            # Read the number of samples from the file, which is used in parsing the rest of the file
            samples = int.from_bytes(audio_file.read(4), sys.byteorder)
            # The rest of the file consists of 2-byte integers, one per channel per sample; read all of it
            raw_data = audio_file.read(CHANNELS * samples * 2)
    # If we are not permitted to load the file or if it does not exist, notify the user and return an empty list with no examples
    except (PermissionError, FileNotFoundError):
        print(f'File {audio_file_path} could not be loaded')
        return []
    # Convert the data into a 1-dimensional NumPy array
    data_array_flat = np.frombuffer(raw_data, dtype=np.int16)
    # Reshape the 1-dimensional array into channels and samples
    data_array = np.reshape(data_array_flat, (samples, CHANNELS))
    # Index and return the data of piezos 3 and 7
    data_array = data_array[:, [piezo_3_index, piezo_7_index]]
    # Cut out most of the empty noise at the beginning, and the hydraulic pump sounds at the end
    data_array = data_array[90_000:190_000]
    # Return the data array without any further processing (temporary)
    return [data_array]


def load_bubble_frequency_domain(bubble: BubbleDataPoint, banded: bool = True) -> List[np.ndarray]:
    """Given a bubble data point, load a flattened representation of the audio in frequency domain, either in various time and frequency bins or continuous, and for both channels; 1 example will be in the returned list, or 0 if it cannot be loaded"""
    # If banding is disabled, and the full resolution data has already been loaded, return it
    if not banded and hasattr(bubble, 'full_resolution_frequency_domain'):
        return bubble.full_resolution_frequency_domain
    # First, try to get the audio waveform corresponding to this bubble
    try:
        time_domain = load_bubble_audio(bubble)[0]
    # If the list is empty, return the same thing
    except IndexError:
        return []
    # Convert the audio into frequency domain
    frequency_domain = time_to_frequency_domain(time_domain)
    # Get the magnitude of the complex outputs
    frequency_magnitudes = np.absolute(frequency_domain)
    # If banding is disabled, return the flattened array of magnitudes directly, wrapped in a single-element list
    if not banded:
        return [frequency_magnitudes.flatten()]
    # Get the frequencies that correspond to the values in the output array, and multiply them by the number of samples per second so they are in Hz
    corresponding_frequencies = np.fft.rfftfreq(time_domain.shape[0]) \
        * SAMPLES_PER_SECOND
    # Create a list to add the resonant energies of the frequency bands to
    resonant_energies = []
    # Iterate over the indices of the frequency band edges, excluding the last one
    for band_edge_index in range(len(FREQUENCY_BANDS) - 1):
        # Get the start and end frequencies of the band corresponding to this index
        start_frequency = FREQUENCY_BANDS[band_edge_index]
        end_frequency = FREQUENCY_BANDS[band_edge_index + 1]
        # Get the indices of samples corresponding to frequencies within this band
        band_indices = np.argwhere(np.logical_and(
            # Include the start frequency but exclude the end frequency so values are not duplicated
            corresponding_frequencies >= start_frequency,
            corresponding_frequencies < end_frequency
        ))
        # Iterate over both piezos
        for piezo_index in range(2):
            # The resonant energy of this frequency band, for this piezo, is the sum of the squares of the frequency magnitudes multiplied by their corresponding frequencies, divided by the squared number of samples in the array
            resonant_energies.append(
                np.sum(
                    (
                        frequency_magnitudes[band_indices, piezo_index]
                        * corresponding_frequencies[band_indices]
                    ) ** 2)
                / (len(corresponding_frequencies) ** 2)
            )
    # Return the resonant energies as a NumPy array, wrapped in a single-element list
    return [np.array(resonant_energies)]


def load_bubble_images(bubble: BubbleDataPoint) -> List[np.ndarray]:
    """Given a bubble data point, load, crop, and return a stacked array of windows (wrapped in a list) which contain that bubble"""
    # If the images have already been combined with the data, just return the ones on the object
    if hasattr(bubble, 'images'):
        return bubble.images
    # Get the path to the bubble data folder and use it to get that to the image folder
    image_folder_path = os.path.join(
        bubble_data_path(bubble),
        'Images'
    )
    # Create a list to hold all images of this bubble
    bubble_images = []
    # Iterate over the number of each of the four cameras and the position of the bubble in this camera
    for camera_number, (bubble_x, bubble_y) in enumerate(bubble.camera_positions):
        # If either axis is less than zero, the bubble could not be found; skip to the next iteration
        if bubble_x < 0 or bubble_y < 0:
            continue
        # Create a list to add the bubbles for this camera to
        camera_bubble_images = []
        # Otherwise, iterate over several frame indices before and after the bubble detection is triggered
        for image_number in range(START_IMAGE_INDEX, END_IMAGE_INDEX):
            # Format the full path of this image
            image_path = os.path.join(
                image_folder_path,
                f'cam{camera_number}_image{image_number}.png'
            )
            # Attempt to load it as a NumPy array, returning from the function if it we are not permitted to load the image or if it does not exist
            try:
                full_image = imread(image_path)
            except (PermissionError, FileNotFoundError):
                print(f'File {image_path} could not be loaded')
                return []
            # Round the bubble X and Y positions to integers and cap them at the edges of the image, so they can be used to index the image
            bubble_x_integer, bubble_y_integer = (
                min(max(int(round(position)), WINDOW_SIDE_LENGTH),
                    shape_dimension - (WINDOW_SIDE_LENGTH + 1))
                for position, shape_dimension in zip([bubble_x, bubble_y], full_image.shape)
            )
            # Crop a square out, centered at the integer position
            half_side_length = WINDOW_SIDE_LENGTH // 2
            window = full_image[(bubble_x_integer - half_side_length):(bubble_x_integer + half_side_length),
                                (bubble_y_integer - half_side_length):(bubble_y_integer + half_side_length)]
            # Add the cropped window to the list of images for this camera
            camera_bubble_images.append(window)
        # Combine the images for this camera into a NumPy array, stacking on the last axis (channels), and add it to the list
        images_array = np.stack(camera_bubble_images, axis=2)
        bubble_images.append(images_array)
    # Return the list of arrays of bubble images for each camera
    return bubble_images
