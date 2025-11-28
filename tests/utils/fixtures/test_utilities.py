import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs


def create_evaluator_args(chromosome_values):
    """
    Create evaluator arguments from chromosome values.

    :param chromosome_values: List or array of chromosome values
    :return: GaEvaluatorArgs instance
    """
    from ..factories.individual_factories import create_individual

    individual = create_individual(chromosome_values)
    return GaEvaluatorArgs(individual)


def evaluate_population(evaluator, population):
    """
    Evaluate an entire population with the given evaluator.

    :param evaluator: Evaluator instance
    :param population: List of GaIndividual instances
    :return: List of fitness values
    """
    fitness_values = []
    for individual in population:
        args = GaEvaluatorArgs(individual)
        fitness = evaluator.evaluate(args)
        fitness_values.append(fitness)
    return fitness_values


def assert_fitness_values(
    fitness_values, expected_count=None, min_value=None, max_value=None
):
    """
    Assert properties of fitness values.

    :param fitness_values: List of fitness values
    :param expected_count: Expected number of values
    :param min_value: Minimum expected value
    :param max_value: Maximum expected value
    """
    if expected_count is not None:
        assert len(fitness_values) == expected_count

    assert all(isinstance(f, (int, float)) for f in fitness_values)

    if min_value is not None:
        assert all(f >= min_value for f in fitness_values)

    if max_value is not None:
        assert all(f <= max_value for f in fitness_values)


def create_test_bounds(dimension, lower=0.0, upper=1.0):
    """
    Create bounds for testing mutation operators.

    :param dimension: Number of dimensions
    :param lower: Lower bound value
    :param upper: Upper bound value
    :return: List of (lower, upper) tuples
    """
    return [(lower, upper) for _ in range(dimension)]


def verify_bounds_compliance(individual, bounds):
    """
    Verify that an individual's chromosome respects the given bounds.

    :param individual: GaIndividual instance
    :param bounds: List of (lower, upper) bound tuples
    :return: True if all values are within bounds
    """
    chromosome = individual.real_chrom
    for i, (lower, upper) in enumerate(bounds):
        if not (lower <= chromosome[i] <= upper):
            return False
    return True


def calculate_diversity(population):
    """
    Calculate standard deviation of each dimension.

    :param population: List of GaIndividual instances
    :return: Array of standard deviations for each dimension
    """
    if not population:
        return np.array([])

    chromosome_matrix = np.array([ind.real_chrom for ind in population])
    return np.std(chromosome_matrix, axis=0)


def find_best_individual(population, fitness_values, minimize=True):
    """
    Find the best individual in a population based on fitness.

    :param population: List of GaIndividual instances
    :param fitness_values: List of corresponding fitness values
    :param minimize: True for minimization problems, False for maximization
    :return: Tuple of (best_individual, best_fitness, best_index)
    """
    if minimize:
        best_index = np.argmin(fitness_values)
    else:
        best_index = np.argmax(fitness_values)

    return population[best_index], fitness_values[best_index], best_index
