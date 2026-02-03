import copy
from typing import List, Sequence, Union

import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class DebGoyalMutation(GaOperator):
    def __init__(
        self,
        distribution_index: int = 20,
        delta_max: Union[float, Sequence[float]] = 0.1,
    ):
        """
        Initializes Deb & Goyal mutation operator for real-valued chromosomes.

        :param distribution_index: Distribution index for mutation (>= 1)
        :type distribution_index: int
        :param delta_max: Maximum perturbation (float or per-gene sequence)
        :type delta_max: float | Sequence[float]
        :raises ValueError: If distribution_index < 1 or delta_max length is invalid
        """
        if int(distribution_index) < 1:
            raise ValueError("distribution_index must be >= 1.")
        self.distribution_index = int(distribution_index)
        self.delta_max = delta_max

    def category(self) -> GaOpCategory:
        """
        Returns operator category.

        :returns: Category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def _delta(self, u: float) -> float:
        """
        Computes Deb & Goyal perturbation coefficient delta in (-1, 1).
        """
        eta = self.distribution_index
        if u < 0.5:
            return (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        return 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Mutates a single randomly chosen gene and clips to domain bounds.

        Assumes mutation probability is handled by the framework.
        """
        population = args.population
        if not population:
            return []

        n_genes = len(population[0].real_chrom)
        domain = args.evaluator.real_domain()

        if isinstance(self.delta_max, (list, tuple, np.ndarray)):
            delta_max_arr = np.asarray(self.delta_max, dtype=float)
            if len(delta_max_arr) != n_genes:
                raise ValueError("delta_max length must match chromosome length.")
        else:
            delta_max_arr = np.full(n_genes, float(self.delta_max), dtype=float)

        new_population: List[GaIndividual] = []

        for individual in population:
            child = copy.deepcopy(individual)

            k = np.random.randint(0, n_genes)
            delta = self._delta(np.random.rand())

            low, up = domain[k]
            new_val = child.real_chrom[k] + delta * delta_max_arr[k]
            child.real_chrom[k] = float(np.clip(new_val, low, up))

            new_population.append(child)

        return new_population
