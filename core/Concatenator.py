from enum import Enum

class Concatenator(Enum):
    """
    Enumeration representing logical concatenators for expressions.

    The available options are AND and OR, which are used to combine conditions
    in logical expressions.
    """
    AND = "AND"
    OR = "OR"
