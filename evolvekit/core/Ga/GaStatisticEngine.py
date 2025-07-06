from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics
import time


class GaStatisticEngine(GaStatistics):

    def start(self, state: GaState):
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
        pass

    def refresh(self, state: GaState):
        pass
