import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual


def population_factory(dimension, population_size, seed=None, individual_type="random"):
    """
    Create a population of individuals with specified dimension and size.

    :param dimension: Number of dimensions for each individual's chromosome
    :param population_size: Number of individuals in the population
    :param seed: Optional random seed for reproducible results
    :param individual_type: Type of individuals to create ("random", "zero", "ones")
    :return: List of GaIndividual instances
    """
    if seed is not None:
        np.random.seed(seed)

    population = []
    for i in range(population_size):
        if individual_type == "random":
            chromosome = np.random.rand(dimension)
        elif individual_type == "zero":
            chromosome = np.zeros(dimension)
        elif individual_type == "ones":
            chromosome = np.ones(dimension)
        else:
            raise ValueError(f"Unknown individual_type: {individual_type}")

        individual = GaIndividual(real_chrom=chromosome)
        population.append(individual)
    return population
