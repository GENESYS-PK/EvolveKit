import copy
import numpy as np
from typing import List, Optional

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class VirusInfectionMutation(GaOperator):
    def __init__(
        self,
        virus_vectors: List[List[Optional[float]]],
        p_copy: float = 0.5,
        p_replace: float = 0.1,
    ):
        """
        Initializes the virus infection mutation operator for
        real-valued chromosomes.

        :param virus_vectors: A list of virus patterns, where each
        virus is a list of float values or None (wildcard indicating
        no overwrite).
        :type virus_vectors: List[List[Optional[float]]]

        :param p_copy: Probability of copying gene from a donor
        individual during virus update
        :type p_copy: float

        :param p_replace: Probability of replacing a virus gene
         with None (wildcard).
        :type p_replace: float
        """
        self.virus_vectors = virus_vectors
        self.life_force = [3] * len(virus_vectors)
        self.p_copy = p_copy
        self.p_replace = p_replace

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify
        its type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued
        mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Applies virus infection mutation operator to a given
        population of individuals.

        Each virus attempts to infect one unused individual by
        overwriting its genes with predefined values from the
        virus vector. After a fixed number of uses (life force),
        each virus can mutate by copying genes from another
        individual or by introducing wildcards (None).

        :param args: GaOperatorArgs containing the current population
        to be mutated
        :type args: GaOperatorArgs
        :returns: A new population list with mutated individuals
        and untouched remainder
        """
        new_population = []
        used = set()
        n = len(self.virus_vectors[0])

        for j, virus in enumerate(self.virus_vectors):
            candidates = [i for i in range(len(args.population)) if i not in used]
            if not candidates:
                break
            chosen = np.random.choice(candidates)
            used.add(chosen)

            infected = copy.deepcopy(args.population[chosen])
            for i in range(n):
                if virus[i] is not None:
                    infected.real_chrom[i] = virus[i]

            new_population.append(infected)

            self.life_force[j] -= 1
            if self.life_force[j] < 0:
                if np.random.rand() < 0.5:
                    donor = np.random.choice(args.population)
                    for i in range(n):
                        if np.random.rand() < self.p_copy:
                            self.virus_vectors[j][i] = donor.real_chrom[i]
                else:
                    for i in range(n):
                        if np.random.rand() < self.p_replace:
                            self.virus_vectors[j][i] = None
                self.life_force[j] = 3

        untouched = [
            copy.deepcopy(ind) for i, ind in enumerate(args.population) if i not in used
        ]
        return new_population + untouched
