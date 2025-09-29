from enum import Enum, auto


class GaAction(Enum):
    """
    Enum representing possible actions for the genetic algorithm process.

    Members:
        CONTINUE: Continue the genetic algorithm execution.
        TERMINATE: Terminate the genetic algorithm execution.
    """

    CONTINUE = auto()
    TERMINATE = auto()
