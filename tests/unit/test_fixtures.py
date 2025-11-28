import numpy as np
import pytest
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator

from tests.fixtures.common_fixtures import (
    create_individual,
    create_evaluator_args,
    evaluate_population,
    assert_fitness_values,
    create_test_bounds,
    verify_bounds_compliance,
    calculate_diversity,
    find_best_individual,
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
