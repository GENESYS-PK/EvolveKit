from evolvekit.core.Ga import GaIndividual, GaStatisticEngine
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy

from typing import List


class GaState:
    current_population: List[GaIndividual]
    selected_population: List[GaIndividual]
    offspring_population: List[GaIndividual]
    elite_population: List[GaIndividual]
    elite_population: List[GaIndividual]
    statistic_engine: GaStatisticEngine
    evaluator: GaEvaluator
    real_clamp_strategy: GaClampStrategy
    crossover_prob: float
    mutation_prob: float
    max_generations: int
    seed: int

    @property
    def population_size(self) -> int:
        return len(self.current_population)

    @property
    def elite_size(self) -> int:
        return len(self.elite_population)
