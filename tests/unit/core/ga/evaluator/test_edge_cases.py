"""
Unit tests for GaEvaluator edge cases and boundary conditions.

Tests extreme scenarios like very large or small values, high dimensions,
and boundary testing.
"""

import numpy as np
import pytest

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.factories.individual_factories import (
    ones_individual_factory,
    uniform_individual_factory,
    create_individual
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_dimension_evaluator(self):
        """Test evaluator with single dimension.
        
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=1)
        individual = create_individual([5.0])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == 25.0
    
    def test_high_dimensional_evaluator(self):
        """Test evaluator with high dimensionality.
        
        100 dimensions of 1^2 = 100.
        
        :returns: None
        :raises: None
        """
        dim = 100
        evaluator = SphereEvaluator(dim=dim)
        individual = ones_individual_factory(dim)
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == float(dim)
    
    def test_evaluator_at_domain_boundaries(self):
        """Test evaluator at domain boundaries.
        
        :returns: None
        :raises: None
        """
        dim = 3
        bounds = (-5.12, 5.12)
        evaluator = SphereEvaluator(dim=dim, bounds=bounds)
        
        individual = uniform_individual_factory(dim, bounds[1])
        args = GaEvaluatorArgs(individual)
        fitness = evaluator.evaluate(args)
        expected = dim * (bounds[1] ** 2)
        assert fitness == pytest.approx(expected, abs=1e-10)
    
    def test_very_small_values(self):
        """Test evaluator with very small values.
        
        Should be very close to zero.
        
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=3)
        individual = create_individual([1e-10, 1e-10, 1e-10])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == pytest.approx(0.0, abs=1e-15)
    
    def test_very_large_values(self):
        """Test evaluator with very large values.
        
        Should be approximately 2e12.
        
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=2)
        individual = create_individual([1e6, 1e6])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        
        assert fitness == pytest.approx(2e12, abs=1e-6)
