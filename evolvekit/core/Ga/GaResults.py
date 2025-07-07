import numpy as np
import numpy.typing as npt

from evolvekit.core.Ga.GaStatistics import GaStatistics


class GaResults:
    real_chrom: npt.NDArray[np.float64]
    bin_chrom: npt.NDArray[np.uint8]
    value: float
    total_generations: int
    total_time: float

    def __init__(self, stats: GaStatistics):
        self.total_generations = stats.generation
        self.total_time = stats.last_time - stats.start_time
        self.real_chrom = np.copy(stats.best_indiv.real_chrom)
        self.bin_chrom = np.copy(stats.best_indiv.bin_chrom)
        self.value = stats.best_indiv.value
