import numpy as np
import pytest

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from evolvekit.benchmarks.RastriginEvaluator import RastriginEvaluator
from evolvekit.benchmarks.RosenbrockEvaluator import RosenbrockEvaluator


@pytest.fixture
def random_individual_2d():
    """Create a random 2D individual."""
    return GaIndividual(real_chrom=np.random.rand(2))


@pytest.fixture
def random_individual_5d():
    """Create a random 5D individual."""
    return GaIndividual(real_chrom=np.random.rand(5))


@pytest.fixture
def random_individual_10d():
    """Create a random 10D individual."""
    return GaIndividual(real_chrom=np.random.rand(10))


@pytest.fixture
def zero_individual_5d():
    """Create a 5D individual at the origin."""
    return GaIndividual(real_chrom=np.zeros(5))


@pytest.fixture
def ones_individual_5d():
    """Create a 5D individual with all ones."""
    return GaIndividual(real_chrom=np.ones(5))


@pytest.fixture
def small_population_5d():
    """Create a small population of 5D individuals."""
    population = []
    for i in range(10):
        chromosome = np.random.rand(5)
        individual = GaIndividual(real_chrom=chromosome)
        population.append(individual)
    return population


@pytest.fixture
def large_population_10d():
    """Create a larger population of 10D individuals."""
    population = []
    for i in range(100):
        chromosome = np.random.rand(10)
        individual = GaIndividual(real_chrom=chromosome)
        population.append(individual)
    return population


@pytest.fixture
def sphere_evaluator_5d():
    """Create a 5D Sphere function evaluator."""
    return SphereEvaluator(dim=5)


@pytest.fixture
def rastrigin_evaluator_5d():
    """Create a 5D Rastrigin function evaluator."""
    return RastriginEvaluator(dim=5)


@pytest.fixture
def rosenbrock_evaluator_5d():
    """Create a 5D Rosenbrock function evaluator."""
    return RosenbrockEvaluator(dim=5)


@pytest.fixture
def all_evaluators_5d():
    """Create all benchmark evaluators for 5D problems."""
    return {
        "sphere": SphereEvaluator(dim=5),
        "rastrigin": RastriginEvaluator(dim=5),
        "rosenbrock": RosenbrockEvaluator(dim=5),
    }


def create_individual(chromosome_values):
    """
    Create an individual with specific chromosome values.

    :param chromosome_values: List or array of chromosome values
    :return: GaIndividual instance
    """
    return GaIndividual(real_chrom=np.array(chromosome_values, dtype=np.float64))


def create_evaluator_args(chromosome_values):
    """
    Create evaluator arguments from chromosome values.

    :param chromosome_values: List or array of chromosome values
    :return: GaEvaluatorArgs instance
    """
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
