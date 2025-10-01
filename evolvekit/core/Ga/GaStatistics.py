from dataclasses import dataclass, field

from evolvekit.core.Ga.GaIndividual import GaIndividual


@dataclass
class GaStatistics:
    """
    Data class responsible for storing statistics about
    current run of genetic algoritm. Serves as a base class
    for :class:`GaStatisticEngine`.
    """

    generation: int = field(default=0)
    stagnation: int = field(default=0)
    mean: float = field(default=0.0)
    median: float = field(default=0.0)
    stdev: float = field(default=0.0)
    best_indiv: GaIndividual | None = field(default=None)
    worst_indiv: GaIndividual | None = field(default=None)
    start_time: float = field(default=0.0)
    last_time: float = field(default=0.0)
