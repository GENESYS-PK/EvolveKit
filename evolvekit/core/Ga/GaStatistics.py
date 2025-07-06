from dataclasses import dataclass

from evolvekit.core.Ga.GaIndividual import GaIndividual


@dataclass
class GaStatistics:
    generation: int
    stagnation: int
    mean: float
    median: float
    stdev: float
    best_individual: GaIndividual
    worst_individual: GaIndividual
    start_time: float
    last_time: float
