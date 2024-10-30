from enum import Enum, verify, UNIQUE

@verify(UNIQUE)
class EventListenerType(Enum):
    """
    Enumeration listing all available types of event listeners.
    """
    BEFORE_FITNESS_FUNCTION = 0
    BEFORE_ELITISM = 1
    BEFORE_SELECTION = 2
    BEFORE_CROSSOVER = 3
    BEFORE_MUTATION = 4
    BEFORE_JOB = 5
    BEFORE_TERMINATE_CHECK = 6
    AFTER_TERMINATE_CHECK = 7