"""
Unit tests for GaEvaluator evaluate method.

Tests that evaluation returns correct fitness values for various
benchmark functions and custom evaluators.
"""

import numpy as np
import pytest

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.mocks.mock_objects import MockEvaluator
from tests.utils.factories.individual_factories import (
    random_individual_factory,
    zero_individual_factory,
    ones_individual_factory,
    create_individual
)
from tests.utils.factories.evaluator_factories import evaluator_factory, all_evaluators_factory


class TestEvaluateMethod:
    """Test that evaluation returns correct fitness values."""
    
    def test_sphere_evaluator_at_optimum(self):
        """Test SphereEvaluator returns 0 at optimum (origin).
        
        :returns: None
        :raises: None
        """
        dim = 5
        evaluator = SphereEvaluator(dim=dim)
        individual = zero_individual_factory(dim)
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == pytest.approx(0.0)
    
    def test_sphere_evaluator_nonzero_point(self):
        """Test SphereEvaluator returns correct value at non-zero point.
        
        Point [1, 2, 3] should give 1^2 + 2^2 + 3^2 = 14.
        
        :returns: None
        :raises: None
        """
        dim = 3
        evaluator = SphereEvaluator(dim=dim)
        individual = create_individual([1.0, 2.0, 3.0])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == pytest.approx(14.0)
    
    def test_evaluation_with_different_sized_chromosomes(self):
        """Test that evaluator correctly handles chromosome size.
        
        Should only use first 'dim' elements: 1^2 + 2^2 + 3^2 = 14.
        
        :returns: None
        :raises: None
        """
        dim = 3
        evaluator = SphereEvaluator(dim=dim)
        individual = create_individual([1.0, 2.0, 3.0, 4.0, 5.0])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == pytest.approx(14.0)
    
    def test_multiple_evaluations(self):
        """Test that multiple evaluations work correctly.
        
        :returns: None
        :raises: None
        """
        dim = 2
        evaluator = SphereEvaluator(dim=dim)
        
        points = [[0.0, 0.0], [2.0, 3.0]]
        expected = [0.0, 13.0]
        
        for point, expected_fitness in zip(points, expected):
            individual = create_individual(point)
            args = GaEvaluatorArgs(individual)
            fitness = evaluator.evaluate(args)
            assert fitness == pytest.approx(expected_fitness)
    
    def test_mock_evaluator_tracking_and_reset(self):
        """Test that MockEvaluator tracks evaluations and resets properly.
        
        :returns: None
        :raises: None
        """
        evaluator = MockEvaluator(dim=5, constant_value=10.0)
        individual = random_individual_factory(5, seed=42)
        args = GaEvaluatorArgs(individual)
        
        assert evaluator.evaluation_count == 0
        
        evaluator.evaluate(args)
        evaluator.evaluate(args)
        assert evaluator.evaluation_count == 2
        assert len(evaluator.evaluation_history) == 2
        
        evaluator.reset()
        assert evaluator.evaluation_count == 0
        assert len(evaluator.evaluation_history) == 0
