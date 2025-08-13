from typing import List, Tuple
import numpy as np

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class SphereEvaluator(GaEvaluator):
    """
    Sphere benchmark function:
    f(x) = sum(x_i^2), optimum at x = 0 (minimum).
    Default per-dimension domain: [-5.12, 5.12]
    """

    def __init__(self, dim: int, bounds: Tuple[float, float] = (-5.12, 5.12)):
        """
        Initialize the Sphere evaluator.

        :param dim: number of dimensions
        :param bounds: (lower, upper) bound for each dimension
        :returns: None
        :raises ValueError: if dim <= 0 or lower >= upper
        """
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if bounds[0] >= bounds[1]:
            raise ValueError("Invalid bounds: lower must be < upper")
        self._dim = dim
        self._bounds = bounds

    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Compute fitness value for the provided chromosome.

        :param args: GaEvaluatorArgs containing real_chrom
        :returns: function value f(x)
        :raises: None
        """
        x = args.real_chrom[: self._dim]
        return float(np.sum(np.square(x)))

    def extremum(self) -> GaExtremum:
        """
        Returns the optimization direction.

        :returns: GaExtremum.MINIMUM
        """
        return GaExtremum.MINIMUM

    def real_domain(self) -> List[Tuple[float, float]]:
        """
        Return domain for each real-valued gene.

        :returns: list of (lower, upper) tuples of length dim
        """
        return [self._bounds] * self._dim