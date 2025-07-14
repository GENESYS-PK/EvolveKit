from typing import List

import numpy as np

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaIndividual import GaIndividual


def generate_random_population(
    evaluator: GaEvaluator,
    population_size: int,
) -> List[GaIndividual]:
    """
    Generate a random population of individuals.

    :param evaluator: instance of a problem-specific evaluator
    :param population_size: number of individuals to generate
    :returns: list of randomly initialized GaIndividual instances
    """
    population: List[GaIndividual] = []

    real_domain = evaluator.real_domain()
    bit_length = evaluator.bin_length()

    for _ in range(population_size):
        individual = GaIndividual()

        if real_domain:
            real_genes = [np.random.uniform(low, high) for low, high in real_domain]
            individual.real_chrom = np.array(real_genes, dtype=np.float64)

        if bit_length > 0:
            bits = np.random.randint(0, 2, size=bit_length, dtype=np.uint8)
            individual.bin_chrom = np.packbits(bits)

        individual.value = 0.0
        population.append(individual)

    return population
