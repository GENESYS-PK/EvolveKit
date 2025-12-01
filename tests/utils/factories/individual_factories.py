import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual


def random_individual_factory(dimension, seed=None):
    """
    Create a random individual with specified dimension.

    :param dimension: Number of dimensions for the chromosome
    :param seed: Optional random seed for reproducible results
    :return: GaIndividual instance with random real chromosome
    """
    if seed is not None:
        np.random.seed(seed)
    return GaIndividual(real_chrom=np.random.rand(dimension))


def zero_individual_factory(dimension):
    """
    Create an individual at the origin with specified dimension.

    :param dimension: Number of dimensions for the chromosome
    :return: GaIndividual instance with zero-valued chromosome
    """
    return GaIndividual(real_chrom=np.zeros(dimension))


def ones_individual_factory(dimension):
    """
    Create an individual with all ones with specified dimension.

    :param dimension: Number of dimensions for the chromosome
    :return: GaIndividual instance with all-ones chromosome
    """
    return GaIndividual(real_chrom=np.ones(dimension))


def uniform_individual_factory(dimension, value):
    """
    Create an individual with all genes set to a specific value. 

    :param dimension: Number of dimensions for the chromosome
    :param value: Value to set for all genes
    :return: GaIndividual instance with uniform gene values
    """
    return GaIndividual(real_chrom=np.full(dimension, value))


def create_individual(chromosome_values):
    """
    Create an individual with specific chromosome values.

    :param chromosome_values: List or array of chromosome values
    :return: GaIndividual instance
    """
    return GaIndividual(real_chrom=np.array(chromosome_values, dtype=np.float64))
