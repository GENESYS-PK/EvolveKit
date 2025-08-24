from typing import List, Tuple
import numpy as np

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class RosenbrockEvaluator(GaEvaluator):
    """
    Rosenbrock benchmark function:
    f(x) = sum_{i=1..n-1} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    Global optimum at x = (1, ..., 1) (minimum).
    Default per-dimension domain: [-2.048, 2.048]
    """

    def __init__(self, dim: int, bounds: Tuple[float, float] = (-2.048, 2.048)):
        """
        Initialize the Rosenbrock evaluator.

        :param dim: number of dimensions (must be >= 2)
        :param bounds: (lower, upper) bound for each dimension
        :returns: None
        :raises ValueError: if dim < 2 or lower >= upper
        """
        if dim < 2:
            raise ValueError("Rosenbrock requires dim to be greater or equal to 2")
        if bounds[0] >= bounds[1]:
            raise ValueError("Invalid bounds: lower must be lower than upper")
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
        x_i = x[:-1]
        x_next = x[1:]
        return float(np.sum(100.0 * (x_next - x_i**2) ** 2 + (1.0 - x_i) ** 2))

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
