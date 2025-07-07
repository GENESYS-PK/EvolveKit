from typing import List
from copy import deepcopy

from evolvekit.core.Ga import GaEvaluator
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from ._internal import CATEGORY_TO_POPULATION_FIELD


class GaOperatorArgs:
    population: List[GaIndividual]
    evaluator: GaEvaluator
    statistics: GaStatistics

    def __init__(self, state: GaState, category: GaOpCategory):
        try:
            attr_name = CATEGORY_TO_POPULATION_FIELD[category]
            self.population = getattr(state, attr_name)
        except KeyError:
            raise ValueError(f"Invalid category provided: {category}")

        self.statistics = deepcopy(state.statistic_engine)
        self.evaluator = state.evaluator
