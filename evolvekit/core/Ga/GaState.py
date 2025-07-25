from typing import List

from evolvekit.core.Ga import GaIndividual, GaStatisticEngine
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy


class GaState:
    """
    Class containing all data about current state of 
    genetic algorithm run.
    """

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
    population_size: int
    elite_size: int
