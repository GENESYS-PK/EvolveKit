import time

from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics


class GaStatisticEngine(GaStatistics):
    """
    Class responsible for manipulating and keeping track of
    various statistics regarding the current run of genetic
    algorithm.
    """

    def start(self, state: GaState):
        """
        Initializes data class structure of :class:`GaStatisticEngine`
        to default values.        

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        """

        self.generation = 1
        self.stagnation = 0
        self.mean = 0
        self.median = 0
        self.stdev = 0
        self.best_indiv = None
        self.worst_indiv = None
        self.start_time = time.process_time()
        self.last_time = self.start_time

    def advance(self, state: GaState):
        """
        Updates statistics AND increases generation number.

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        """
        pass

    def refresh(self, state: GaState):
        """
        Updates statistics WITHOUT increasing generation number.

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        """
        pass
