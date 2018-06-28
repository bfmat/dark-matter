"""A function for checking that the number of command line arguments is correct, and if it is not, printing an error message and quitting"""
# Created by Brendon Matusch, June 2018

import sys
from typing import List


def verify_arguments(*argument_descriptions: str) -> None:
    """Check that the number of command line arguments is correct, and if it is not, print an error message and quit"""
    # If the length of the list of arguments (minus one because the command is included) is not the same as the number of argument descriptions
    if len(sys.argv) - 1 != len(argument_descriptions):
        # Add angle brackets around the argument descriptions so they are clearly separated
        separated_argument_descriptions = [
            '<' + description + '>' for description in argument_descriptions]
        # Exit, outputting the script command and the expected arguments; they will automatically be separated by spaces
        print('Usage:', sys.argv[0], *separated_argument_descriptions)
        sys.exit()
