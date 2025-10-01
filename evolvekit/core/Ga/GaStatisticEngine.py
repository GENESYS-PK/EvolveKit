from __future__ import annotations
import time
import copy

import numpy as np

from evolvekit.core.Ga import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class GaStatisticEngine(GaStatistics):
    """
    Class responsible for manipulating and keeping track of
    various statistics regarding the current run of genetic
    algorithm.
    """

    def start(self, state: "GaState"):
        """
        Initializes data class structure of :class:`GaStatisticEngine`
        to default values.        

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        :returns: None.
        """
        
        self.generation = 0
        self.stagnation = 0
        self.mean = 0
        self.median = 0
        self.stdev = 0
        self.best_indiv = None
        self.worst_indiv = None
        self.start_time = time.process_time()
        self.last_time = self.start_time

    def advance(self, state: "GaState"):
        """
        Updates statistics AND increases generation number.

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        :returns: None.
        """

        prev_best = self.best_indiv
        self.refresh(state)

        self.generation += 1
        if prev_best and prev_best.value == self.best_indiv.value:
            self.stagnation += 1
        else:
            self.stagnation = 0

        self.last_time = time.process_time()

    def refresh(self, state: "GaState"):
        """
        Updates statistics WITHOUT increasing generation number.

        :param state: Object representing current state of evolution loop.
        :type state: :class:`GaState`.
        :returns: None.
        """

        all_values = np.array([ind.value for ind in state.current_population])

        self.mean = np.mean(all_values)
        self.median = np.median(all_values)
        self.stdev = np.std(all_values)

        key_func = lambda ind: ind.value
        extremum = state.evaluator.extremum()
        population = state.current_population

        match extremum:
            case GaExtremum.MAXIMUM:
                self.best_indiv = copy.deepcopy(max(population, key=key_func))
                self.worst_indiv = copy.deepcopy(min(population, key=key_func))
            case GaExtremum.MINIMUM:
                self.best_indiv = copy.deepcopy(min(population, key=key_func))
                self.worst_indiv = copy.deepcopy(max(population, key=key_func))
