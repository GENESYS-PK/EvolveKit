from typing import List, Tuple
import numpy as np

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class RastriginEvaluator(GaEvaluator):
    """
    Rastrigin benchmark function:
    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)), A = 10
    Global optimum at x = 0 (minimum).
    Default per-dimension domain: [-5.12, 5.12]
    """

    def __init__(self, dim: int, bounds: Tuple[float, float] = (-5.12, 5.12), A: float = 10.0):
        """
        Initialize the Rastrigin evaluator.

        :param dim: number of dimensions
        :param bounds: (lower, upper) bound for each dimension
        :param A: function parameter (default 10.0)
        :returns: None
        :raises ValueError: if dim <= 0, lower >= upper, or A <= 0
        """
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if bounds[0] >= bounds[1]:
            raise ValueError("Invalid bounds: lower must be < upper")
        if A <= 0:
            raise ValueError("A must be > 0")
        self._dim = dim
        self._bounds = bounds
        self._A = A

    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Compute fitness value for the provided chromosome.

        :param args: GaEvaluatorArgs containing real_chrom
        :returns: function value f(x)
        :raises: None
        """
        x = args.real_chrom[: self._dim]
        return float(self._A * self._dim + np.sum(x**2 - self._A * np.cos(2 * np.pi * x)))

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