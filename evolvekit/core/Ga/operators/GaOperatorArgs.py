from typing import List

from evolvekit.core.Ga import GaEvaluator
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory

CATEGORY_TO_POPULATION_FIELD = {
    GaOpCategory.SELECTION: "current_population",
    GaOpCategory.REAL_CROSSOVER: "selected_population",
    GaOpCategory.REAL_MUTATION: "offspring_population",
    GaOpCategory.BIN_CROSSOVER: "selected_population",
    GaOpCategory.BIN_MUTATION: "offspring_population",
}


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

        self.statistics = state.statistic_engine
        self.evaluator = state.evaluator
