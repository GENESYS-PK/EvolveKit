from typing import List
from copy import deepcopy

from evolvekit.core.Ga import GaEvaluator
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from ._internal import CATEGORY_TO_POPULATION_FIELD


class GaOperatorArgs:
    """
    Class representing arguments required by genetic algorithm operators.

    Attributes:
        population (List[GaIndividual]): Object that stores a list of individuals.
            The type of population depends on the context:
            for the selection algorithm, it will store the current population;
            for crossover, the population of selected individuals;
            and for mutation, the population of offspring.
            This object will hold a deep copy, so even if the operator modifies it,
            it should not affect the state of evolution.
        evaluator (GaEvaluator): Object that stores a reference to the GaEvaluator class instance
            (i.e., we do not create a copy, just assign the reference).
            This allows each operator to access the evaluation function
            without having to manually pass it in the constructor.
        statistics (GaStatistics): Object that stores general statistics such as
            the current generation number, best/worst individual, etc. A deep copy is created,
            so if the operator changes the statistics,
            it will not affect the "real" statistics in the evolution state.
    """
    population: List[GaIndividual]
    evaluator: GaEvaluator
    statistics: GaStatistics

    def __init__(self, state: GaState, category: GaOpCategory):
        """
        The constructor allows for quick creation of a GaOperatorArgs object based on the evolution state.
        The `category` argument is required because the `population` member object depends on the context, i.e.,
        depending on the operator category, a different population is passed.

        :param state: The current state of the genetic algorithm.
        :type state: GaState
        :param category: The operator category.
        :type category: GaOpCategory
        :raises ValueError: If an invalid category is provided.
        """
        try:
            attr_name = CATEGORY_TO_POPULATION_FIELD[category]
            self.population = getattr(state, attr_name)
        except KeyError:
            raise ValueError(f"Invalid category provided: {category}")

        self.statistics = deepcopy(state.statistic_engine)
        self.evaluator = state.evaluator
