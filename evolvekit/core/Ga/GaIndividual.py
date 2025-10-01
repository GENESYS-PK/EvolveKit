from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class GaIndividual:
    """
    Class representing a single, potential solution of a problem
    posited in the simulation.
    """

    real_chrom: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    bin_chrom: npt.NDArray[np.uint8] = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )
    value: float = field(default=0.0)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}

        copy = GaIndividual(
            real_chrom=np.copy(self.real_chrom),
            bin_chrom=np.copy(self.bin_chrom),
            value=self.value,
        )

        memodict[id(self)] = copy
        return copy
