from typing import List, Union
import numpy as np
import copy

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class VirusInfectionMutation(GaOperator):
    def __init__(self, virus_vectors: List[List[Union[int, str]]], p_copy: float = 0.5,
                 p_replace: float = 0.1):
        """
        Initializes the virus infection mutation operator with given
        virus vectors and mutation parameters.

        :param virus_vectors: A list of virus patterns, where each
        virus is a list of 0, 1, or '*'
        :type virus_vectors: List[List[Union[int, str]]]

        :param p_copy: Probability of copying gene from a donor
        individual during virus update
        :type p_copy: float

        :param p_replace: Probability of replacing gene with
        wildcard '*' during virus reset
        :type p_replace: float
        """
        self.virus_vectors = virus_vectors
        self.life_force = [3] * len(virus_vectors)
        self.p_copy = p_copy
        self.p_replace = p_replace

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its type
        in the evolutionary algorithm framework.

        :returns: The operator category indicating binary mutation
        """
        return GaOpCategory.BIN_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Applies virus infection mutation operator to a given
        population of individuals.

        This operator infects selected individuals using predefined
        virus vectors by modifying bits in their binary chromosomes.
        Each virus has a limited life span (`life_force`). After it
        expires, the virus is either updated using bits from a random
        donor individual or partially reset to wildcards (`'*'`).

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
            bits = np.unpackbits(infected.bin_chrom)

            for i in range(n):
                if virus[i] == 0:
                    bits[i] = 0
                elif virus[i] == 1:
                    bits[i] = 1
                elif virus[i] == '*':
                    pass

            infected.bin_chrom = np.packbits(bits)
            new_population.append(infected)

            self.life_force[j] -= 1
            if self.life_force[j] < 0:
                if np.random.rand() < 0.5:
                    donor = np.random.choice(args.population)
                    donor_bits = np.unpackbits(donor.bin_chrom)
                    for i in range(n):
                        if np.random.rand() < self.p_copy:
                            self.virus_vectors[j][i] = int(donor_bits[i])
                else:
                    for i in range(n):
                        if np.random.rand() < self.p_replace:
                            self.virus_vectors[j][i] = '*'
                self.life_force[j] = 3

        untouched = [copy.deepcopy(ind) for i, ind in enumerate(args.population)
                     if i not in used]
        return new_population + untouched
