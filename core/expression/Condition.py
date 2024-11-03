from abc import ABC, abstractmethod
from typing import Self


class Condition(ABC):
    """
    Abstract base class for conditions in an evolution process.

    This class should be inherited by concrete conditions that define
    specific criteria for evaluation during evolution.
    """

    def __init__(self):
        """
        Initialize the Condition base class.
        """
        pass

    @abstractmethod
    def evaluate(self, evolution: Self) -> bool:
        """
        Evaluate the condition based on the provided evolution instance.

        :param evolution: The evolution process instance to evaluate.
        :return: A boolean indicating whether the condition is met.
        """
        pass
