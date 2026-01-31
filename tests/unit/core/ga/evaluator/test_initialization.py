"""
Unit tests for GaEvaluator initialization.

Tests proper setup of evaluators with different parameters,
including dimension validation and bounds checking.
"""

import pytest

from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.mocks.mock_objects import MockEvaluator
from tests.utils.factories.evaluator_factories import evaluator_factory, all_evaluators_factory


class TestEvaluatorInitialization:
    """Test proper setup of evaluators with different parameters."""
    
    def test_sphere_evaluator_initialization_default(self):
        """Test SphereEvaluator initializes with default bounds.
        
        :returns: None
        :raises: None
        """
        dim = 5
        evaluator = SphereEvaluator(dim=dim)
        domain = evaluator.real_domain()
        
        assert len(domain) == dim
        assert all(bounds == (-5.12, 5.12) for bounds in domain)
        assert evaluator.extremum() == GaExtremum.MINIMUM
    
    def test_sphere_evaluator_initialization_custom_bounds(self):
        """Test SphereEvaluator initializes with custom bounds.
        
        :returns: None
        :raises: None
        """
        dim = 3
        custom_bounds = (-10.0, 10.0)
        evaluator = SphereEvaluator(dim=dim, bounds=custom_bounds)
        domain = evaluator.real_domain()
        
        assert len(domain) == dim
        assert all(bounds == custom_bounds for bounds in domain)
    
    def test_mock_evaluator_initialization(self):
        """Test MockEvaluator initializes with custom parameters.
        
        :returns: None
        :raises: None
        """
        dim = 8
        constant_value = 42.0
        evaluator = MockEvaluator(dim=dim, constant_value=constant_value)
        
        assert evaluator.dim == dim
        assert evaluator.constant_value == constant_value
        assert evaluator.evaluation_count == 0
    
    def test_invalid_dimension_raises_error(self):
        """Test that invalid dimension raises ValueError.
        
        :returns: None
        :raises ValueError: When dimension is zero or negative
        """
        with pytest.raises(ValueError, match="dim must be greater than 0"):
            SphereEvaluator(dim=0)
        
        with pytest.raises(ValueError, match="dim must be greater than 0"):
            SphereEvaluator(dim=-5)
    
    def test_invalid_bounds_raises_error(self):
        """Test that invalid bounds raise ValueError.
        
        :returns: None
        :raises ValueError: When bounds are invalid
        """
        with pytest.raises(ValueError, match="Invalid bounds"):
            SphereEvaluator(dim=5, bounds=(10.0, -10.0))
        
        with pytest.raises(ValueError, match="Invalid bounds"):
            SphereEvaluator(dim=5, bounds=(5.0, 5.0))
