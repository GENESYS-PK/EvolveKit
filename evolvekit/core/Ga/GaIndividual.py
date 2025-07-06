from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class GaIndividual:
    real_chrom: npt.NDArray[np.float64]
    bin_chrom: npt.NDArray[np.uint8]
    value: float
