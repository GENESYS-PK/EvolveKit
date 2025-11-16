from abc import ABC, abstractmethod
from typing import List, Tuple

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class GaEvaluator(ABC):
    """
    Abstract class representing fitness function.
    """

    @abstractmethod
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Calculates the fitness value for a given set of arguments.

        :param args: Object representing a particular solution for the
            posited problem.
        :returns: A value representing fitness for this particular
            solution.
        :rtype: float
        """
        pass

    @abstractmethod
    def extremum(self) -> GaExtremum:
        """
        Returns optimization criterion for the posited problem.

        :returns: A value signaling whether to search for minimum
            or maximum in posited problem.
        """
        pass

    def real_domain(self) -> List[Tuple[float, float]]:
        """
        Returns the domain of the real valued chromosome.
        If return value is set to an empty list, real valued
        chromosome is ignored.

        :returns: A list of tuples representing lower and
            upper bounds, respectively, of every gene domain.
            Defaults to an empty list.
        :rtype: list
        """

        return []

    def bin_length(self) -> int:
        """
        Returns bit length of the bit-string represented chromosome.
        If return value is set to 0, bit-string represented
        chromosome is ignored. Returned value must be nonnegative.

        :returns: A length in bits. Defaults to 0.
        :rtype: int
        """
        return 0
