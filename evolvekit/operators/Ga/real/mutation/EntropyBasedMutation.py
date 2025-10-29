from typing import List
import math
import sys

import numpy as np

from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class EntropyBasedMutation(GaOperator):
    def __init__(self, L: int = 8):
        """
        Initializes uniform mutation operator for real-valued
        chromosomes.

        :param L: number of intervals or number of bins used for
            discretizing the fitness range.
        :type L: int
        """
        self.L = L

    def category(self) -> GaOpCategory:
        """
        Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs entropy based muttation on real-valued chromosomes
        in the population.

        :param args: Container with population and evaluator for
            mutation operation
        :type args: GaOperatorArgs
        :returns: Population with potentially mutated individuals
        :rtype: List[GaIndividual]
        """
        population = args.population
        for individual in population:
            individual.value = args.evaluator.evaluate(GaEvaluatorArgs(individual))
        max_evauluation = max(population, key=lambda x: x.value).value
        min_evaulation = min(population, key=lambda x: x.value).value

        calcualted_range = (max_evauluation - min_evaulation) / self.L
        number_of_occurances = sum(
            map(
                lambda x: x.value >= min_evaulation and x.value < calcualted_range,
                population,
            )
        )
        ranges = [[min_evaulation, calcualted_range, number_of_occurances]]
        for i in range(1, self.L - 1):
            inclusive_start = i * calcualted_range
            exclusive_end = (i + 1) * calcualted_range
            number_of_occurances = sum(
                map(
                    lambda x: x.value >= inclusive_start and x.value < exclusive_end,
                    population,
                )
            )
        inclusive_start = i * calcualted_range
        exclusive_end = (i + 1) * max_evauluation + sys.float_info.epsilon
        number_of_occurances = sum(
            map(
                lambda x: x.value >= inclusive_start and x.value < exclusive_end,
                population,
            )
        )
        ranges.append([inclusive_start, exclusive_end, number_of_occurances])
        relative_occurences = []
        size_of_population = len(population)
        for interval in ranges:
            relative_occurences.append(interval[2] / size_of_population)
        result_of_sum = 0
        for interval in ranges:
            if interval[2] != 0:
                result_of_sum += interval[2] * math.log(interval[2])
        h = (1 / math.log(size_of_population)) * result_of_sum
        p_c = max((0.25, h))
        p_ebm = (1 - math.sqrt(p_c)) / 2
        for i in range(len(individual.real_chrom)):
            if np.random.uniform(low=0.0, high=1 + sys.float_info.epsilon) < p_ebm:
                r = np.random.normal(loc=0, scale=h**2)
                individual.real_chrom[i] += r
        return population
