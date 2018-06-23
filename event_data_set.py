"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

# The path to the main data file which contains processed attributes of bubble events
EVENT_FILE_PATH = "/opt/merged_all_all.txt"


class EventDataSet:
    """A bubble event data set class that is loaded from CERN ROOT data as well as audio recordings and images, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    def __init__(self) -> None:
        """Initializer that takes parameters that determine which data is loaded"""
        # Open the event file and read its full contents
        with open(EVENT_FILE_PATH) as event_file:
            event_file_lines = event_file.readlines()
        # The second line of the file contains the names of the various fields, separated by spaces
        field_names = event_file_lines[1].split()
