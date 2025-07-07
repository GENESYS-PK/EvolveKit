from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class GaIndividual:
    real_chrom: npt.NDArray[np.float64]
    bin_chrom: npt.NDArray[np.uint8]
    value: float

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
