import random
from typing import List

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaStatisticEngine import GaStatisticEngine
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy


class GaState:
    """
    Class containing all data about current state of
    the genetic algorithm.
    """

    current_population: List[GaIndividual]
    selected_population: List[GaIndividual]
    offspring_population: List[GaIndividual]
    elite_population: List[GaIndividual]
    statistic_engine: GaStatisticEngine
    evaluator: GaEvaluator | None
    real_clamp_strategy: GaClampStrategy
    crossover_prob: float
    mutation_prob: float
    max_generations: int
    seed: int
    population_size: int
    elite_size: int

    def __init__(self):
        self.current_population = []
        self.selected_population = []
        self.offspring_population = []
        self.elite_population = []
        self.evaluator = None
        self.real_clamp_strategy = GaClampStrategy.NONE
        self.crossover_prob = 0
        self.mutation_prob = 0
        self.max_generations = 0
        self.seed = random.randint(1, 2**32 - 1)
        self.population_size = 0
        self.elite_size = 0
        self.statistic_engine = GaStatisticEngine()
