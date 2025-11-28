"""
Comprehensive tests for fixture utility functions and factory-based fixtures.

This module tests both the utility helper functions (like create_individual,
evaluate_population, etc.) and demonstrates best practices for using the
refactored fixture factories with pytest parametrization.
"""

import numpy as np
import pytest
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator

from tests.utils import (
    create_individual,
    create_evaluator_args,
    evaluate_population,
    assert_fitness_values,
    create_test_bounds,
    verify_bounds_compliance,
    calculate_diversity,
    find_best_individual,
    random_individual_factory,
    zero_individual_factory,
    ones_individual_factory,
    uniform_individual_factory,
    population_factory,
    evaluator_factory,
    all_evaluators_factory,
)


def test_create_individual():
    """
    Test creating individuals with specific chromosome values.

    Creates a GaIndividual with real chromosome [1.0, 2.0, 3.0]
    and verifies the instance type and chromosome values.

    :returns: None
    :raises: None
    """
    individual = create_individual([1.0, 2.0, 3.0])
    assert isinstance(individual, GaIndividual)
    np.testing.assert_array_equal(individual.real_chrom, [1.0, 2.0, 3.0])


def test_create_evaluator_args():
    """
    Test creating evaluator arguments from chromosome values.

    Creates GaEvaluatorArgs from chromosome [1.0, 2.0] and verifies
    the args object has real_chrom attribute with correct values.

    :returns: None
    :raises: None
    """
    args = create_evaluator_args([1.0, 2.0])
    assert args is not None
    assert hasattr(args, "real_chrom")
    np.testing.assert_array_equal(args.real_chrom, [1.0, 2.0])


def test_evaluate_population():
    """
    Test population evaluation utility function.

    Evaluates a population of 2 individuals using SphereEvaluator:
    - Individual at [0.0, 0.0] should have fitness 0.0
    - Individual at [1.0, 1.0] should have fitness 2.0 (1^2 + 1^2)

    :returns: None
    :raises: None
    """
    evaluator = SphereEvaluator(dim=2)
    population = [create_individual([0.0, 0.0]), create_individual([1.0, 1.0])]
    fitness_values = evaluate_population(evaluator, population)
    assert len(fitness_values) == 2
    assert fitness_values[0] == 0.0  # Sphere at origin
    assert fitness_values[1] == 2.0  # Sphere at [1,1]


def test_assert_fitness_values():
    """
    Test fitness validation utility function.

    Tests assert_fitness_values with valid parameters and verifies
    it raises AssertionError when expected_count doesn't match.

    :returns: None
    :raises: None
    """
    fitness_values = [1.0, 2.0, 3.0]

    assert_fitness_values(
        fitness_values, expected_count=3, min_value=0.0, max_value=5.0
    )

    with pytest.raises(AssertionError):
        assert_fitness_values(fitness_values, expected_count=2)


def test_create_test_bounds():
    """
    Test bounds creation utility function.

    Creates bounds for 3 dimensions with range [-1.0, 1.0]
    and verifies all bounds are correctly set.

    :returns: None
    :raises: None
    """
    bounds = create_test_bounds(3, -1.0, 1.0)
    assert len(bounds) == 3
    assert all(b == (-1.0, 1.0) for b in bounds)


def test_verify_bounds_compliance():
    """
    Test bounds verification utility function.

    Tests verify_bounds_compliance with:
    - Individual [0.5, -0.5] within bounds [-1.0, 1.0] (should pass)
    - Individual [2.0, 0.0] outside bounds [-1.0, 1.0] (should fail)

    :returns: None
    :raises: None
    """
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    individual_ok = create_individual([0.5, -0.5])
    assert verify_bounds_compliance(individual_ok, bounds) == True

    individual_bad = create_individual([2.0, 0.0])
    assert verify_bounds_compliance(individual_bad, bounds) == False


def test_calculate_diversity():
    """
    Test diversity calculation utility function.

    Calculates diversity for a population of 3 individuals
    and verifies diversity array length and non-negative values.

    :returns: None
    :raises: None
    """
    population = [
        create_individual([0.0, 0.0]),
        create_individual([1.0, 1.0]),
        create_individual([0.5, 0.5]),
    ]
    diversity = calculate_diversity(population)
    assert len(diversity) == 2
    assert all(d >= 0 for d in diversity)


def test_find_best_individual():
    """
    Test finding best individual utility function.

    Tests find_best_individual with fitness values [0.0, 2.0, 8.0]:
    - For minimization: best should be index 0 with fitness 0.0
    - For maximization: best should be index 2 with fitness 8.0

    :returns: None
    :raises: None
    """
    population = [
        create_individual([0.0, 0.0]),
        create_individual([1.0, 1.0]),
        create_individual([2.0, 2.0]),
    ]
    fitness_values = [0.0, 2.0, 8.0]

    _, best_fit, best_idx = find_best_individual(
        population, fitness_values, minimize=True
    )
    assert best_idx == 0
    assert best_fit == 0.0

    _, best_fit, best_idx = find_best_individual(
        population, fitness_values, minimize=False
    )
    assert best_idx == 2
    assert best_fit == 8.0


# =============================================================================
# Tests for factory-based fixtures
# =============================================================================


@pytest.mark.parametrize("dimension", [2, 5, 10, 20])
def test_individual_factories_with_dimensions(dimension):
    """Test individual factories across different dimensions."""
    # Test random individual
    random_ind = random_individual_factory(dimension, seed=42)
    assert len(random_ind.real_chrom) == dimension
    assert random_ind.real_chrom.dtype == np.float64

    # Test zero individual
    zero_ind = zero_individual_factory(dimension)
    assert len(zero_ind.real_chrom) == dimension
    assert np.allclose(zero_ind.real_chrom, 0.0)

    # Test ones individual
    ones_ind = ones_individual_factory(dimension)
    assert len(ones_ind.real_chrom) == dimension
    assert np.allclose(ones_ind.real_chrom, 1.0)

    # Test uniform individual
    uniform_ind = uniform_individual_factory(dimension, 0.5)
    assert len(uniform_ind.real_chrom) == dimension
    assert np.allclose(uniform_ind.real_chrom, 0.5)


@pytest.mark.parametrize(
    "dimension,population_size",
    [
        (2, 5),
        (5, 10),
        (10, 20),
        (15, 50),
    ],
)
def test_population_factory_with_parameters(dimension, population_size):
    """Test population factory with different dimensions and sizes."""
    population = population_factory(
        dimension=dimension,
        population_size=population_size,
        seed=42,
        individual_type="random",
    )

    assert len(population) == population_size
    for individual in population:
        assert len(individual.real_chrom) == dimension
        assert individual.real_chrom.dtype == np.float64


@pytest.mark.parametrize(
    "evaluator_type,dimension",
    [
        ("sphere", 2),
        ("rastrigin", 5),
        ("rosenbrock", 10),
        ("sphere", 20),
    ],
)
def test_evaluator_factory_with_parameters(evaluator_type, dimension):
    """Test evaluator factory with different types and dimensions."""
    evaluator = evaluator_factory(evaluator_type, dimension)

    # Test that evaluator works with appropriate dimension
    individual = zero_individual_factory(dimension)
    args = GaEvaluatorArgs(individual)
    fitness = evaluator.evaluate(args)

    assert isinstance(fitness, (int, float))

    # Test domain properties
    domain = evaluator.real_domain()
    assert len(domain) == dimension


def test_all_evaluators_factory():
    """Test creating all evaluators at once."""
    dimension = 5
    evaluators = all_evaluators_factory(dimension)

    expected_types = {"sphere", "rastrigin", "rosenbrock"}
    assert set(evaluators.keys()) == expected_types

    # Test each evaluator
    individual = zero_individual_factory(dimension)
    args = GaEvaluatorArgs(individual)

    for eval_type, evaluator in evaluators.items():
        fitness = evaluator.evaluate(args)
        assert isinstance(fitness, (int, float))


def test_population_types():
    """Test different population individual types."""
    dimension = 3
    population_size = 5

    # Test random population
    random_pop = population_factory(
        dimension, population_size, seed=42, individual_type="random"
    )
    assert all(0 <= val <= 1 for ind in random_pop for val in ind.real_chrom)

    # Test zero population
    zero_pop = population_factory(dimension, population_size, individual_type="zero")
    assert all(val == 0 for ind in zero_pop for val in ind.real_chrom)

    # Test ones population
    ones_pop = population_factory(dimension, population_size, individual_type="ones")
    assert all(val == 1 for ind in ones_pop for val in ind.real_chrom)


def test_fixture_fixtures(
    individual_factory, population_factory_fixture, evaluator_factory_fixture
):
    """Test the fixture versions of the factories."""
    # Using fixture that returns factory function
    individual = individual_factory(5, seed=42)
    assert len(individual.real_chrom) == 5

    population = population_factory_fixture(3, 10, seed=42)
    assert len(population) == 10
    assert all(len(ind.real_chrom) == 3 for ind in population)

    evaluator = evaluator_factory_fixture("sphere", 4)
    assert evaluator.real_domain() is not None


# =============================================================================
# Integration tests combining utilities and factories
# =============================================================================


@pytest.mark.parametrize("dimension", [3, 7, 12])
def test_factory_utility_integration(dimension):
    """Test integration between factory functions and utility functions."""
    # Create test data using factories
    population = population_factory(dimension, 20, seed=42, individual_type="random")
    evaluator = evaluator_factory("sphere", dimension)

    # Use utility functions with factory-created data
    fitness_values = evaluate_population(evaluator, population)

    # Validate using utility functions
    assert_fitness_values(fitness_values, expected_count=20, min_value=0.0)

    # Test diversity calculation
    diversity = calculate_diversity(population)
    assert len(diversity) == dimension
    assert all(d >= 0 for d in diversity)

    # Test finding best individual
    best_individual, best_fitness, best_idx = find_best_individual(
        population, fitness_values
    )
    assert 0 <= best_idx < len(population)
    assert best_fitness == fitness_values[best_idx]

    # Test bounds compliance with factory-created bounds
    bounds = create_test_bounds(dimension, 0.0, 1.0)
    for individual in population:
        assert verify_bounds_compliance(individual, bounds)
