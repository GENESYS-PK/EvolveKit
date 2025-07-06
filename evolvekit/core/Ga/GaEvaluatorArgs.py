from evolvekit.core.Ga import GaIndividual
import numpy as np
import numpy.typing as npt


class GaEvaluatorArgs:
    real_chrom: npt.NDArray[np.float64]
    bin_chrom: npt.NDArray[np.uint8]

    def __init__(self, individual: GaIndividual):
        self.real_chrom = np.copy(individual.real_chrom)
        self.bin_chrom = np.copy(individual.bin_chrom)
