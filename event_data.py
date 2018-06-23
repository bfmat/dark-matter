"""Code for managing data sets that allows loading and conversion"""
# Created by Brendon Matusch, June 2018

import ROOT

# The path to the main data file which contains processed attributes of bubble events
EVENT_FILE_PATH = "/opt/merged_all_all.root"


class EventDataSet:
    """A bubble event data set class that is loaded from CERN ROOT data as well as audio recordings and images, and is convertible to many different formats, containing varying data types, that can be used to train neural networks"""

    def __init__(self) -> None:
        """Initializer that takes parameters that determine which data is loaded"""
        # Open the event file and get the main tree
        # These cannot be in the same line or a segmentation fault will occur
        event_file = ROOT.TFile(EVENT_FILE_PATH)
        tree = event_file.Get('T')
        # Iterate over the tree and get a list of the events
        events = [event for event in tree]
