from enum import Enum


class ExitCode(int, Enum):
    """
    Enumeration of standard exit codes.
    Inherits from 'int' to be directly usable with 'sys.exit()'.
    """

    #: Successful termination
    SUCCESS: int = 0
    #: Generic error (catch-all)
    GENERAL_ERROR: int = 1
    #: A specified input file was not found
    TRAINING_DATA_NOT_FOUND: int = 30
    VALIDATION_DATA_NOT_FOUND: int = 31
    #: A specified configuration file was invalid
    CONFIG_VALIDATION_ERROR: int = 70
    CONFIG_FILE_NOT_FOUND: int = 71
