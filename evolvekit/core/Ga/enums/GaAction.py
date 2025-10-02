from enum import Enum, auto


class GaAction(Enum):
    """
    Enum represents possible actions for the genetic algorithm process.

    Inspector returns one of these values. Each value from this enum
    indicates what genetic algorithm should do next.

    :cvar CONTINUE: Continue the genetic algorithm execution.
    :cvar TERMINATE: Terminate the genetic algorithm execution.
    """

    CONTINUE = auto()
    TERMINATE = auto()
