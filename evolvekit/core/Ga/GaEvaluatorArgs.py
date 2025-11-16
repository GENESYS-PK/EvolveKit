import numpy as np
import numpy.typing as npt

from evolvekit.core.Ga import GaIndividual


class GaEvaluatorArgs:
    """
    Class for supplying arguments to :class:`GaEvaulator` class.
    By default contains real and binary chromosome.
    """

    real_chrom: npt.NDArray[np.float64]
    bin_chrom: npt.NDArray[np.uint8]

    def __init__(self, individual: GaIndividual):
        """
        Constructor method.
        Initialize members of :class:`GaEvaluatorArgs`.

        :param individual: Individual to copy data from.
        :type individual: :class:`GaIndividual`.
        :returns: None.
        """
        self.real_chrom = np.copy(individual.real_chrom)
        self.bin_chrom = np.copy(individual.bin_chrom)
