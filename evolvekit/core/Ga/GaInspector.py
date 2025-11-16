from abc import ABC, abstractmethod

from evolvekit.core.Ga.enums.GaAction import GaAction
from evolvekit.core.Ga.GaStatistics import GaStatistics


class GaInspector(ABC):
    """
    Abstract class used to manipulate and inspect genetic algorithm's
    current run.

    To use it, derive your own class from it and implement its methods.
    Next, pass an instantiated object into :class:`GaIsland`.
    """

    def initialize(self):
        """
        Runs at the beginning of simulation, BEFORE first evolution takes place.

        Use this method to initialize services needed by
        the :func:`inspect()` function.
        :returns: None.
        """

        pass

    @abstractmethod
    def inspect(self, stats: GaStatistics) -> GaAction:
        """
        Analyzes current state of simulation and decides on it.

        Use this method to monitor interesting simulation statistics,
        to log various data or to prematurely end simulation upon reaching
        specified criterion.

        :param stats: Object containing statistics of current simulation.
        :type stats: :class:`GaStatistics`.
        :returns: A value representing action chosen after inspection.
        :rtype: :class:`GaAction`.
        """

        pass

    def finish(self, stats: GaStatistics):
        """
        Runs at the end of the simulation.

        Use this method to stop services created by :func:`initialize()`
        method and to log final results.

        :param stats: Object containing statistics of current simulation.
        :type stats: :class:`GaStatistics`.
        :returns: None.
        """

        pass
