"""
Unit tests for GaEvaluator domain methods.

Tests that real_domain() and bin_length() methods correctly
define the search space for different evaluators.
"""

import pytest

from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.mocks.mock_objects import MockEvaluator


class TestRealDomainMethod:
    """Test that domain bounds are correctly set."""
    
    def test_real_domain_length_matches_dimension(self):
        """Test domain list has correct length.
        
        :returns: None
        :raises: None
        """
        dim = 7
        evaluator = SphereEvaluator(dim=dim)
        domain = evaluator.real_domain()
        
        assert len(domain) == dim
    
    @pytest.mark.parametrize("bounds,dim", [
        ((-5.12, 5.12), 3),
        ((-10.0, 10.0), 5),
        ((0.0, 1.0), 2),
    ])
    def test_real_domain_different_bounds(self, bounds, dim):
        """Test domain with different bound values.
        
        :param bounds: Tuple of (lower, upper) bounds
        :param dim: Number of dimensions
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=dim, bounds=bounds)
        domain = evaluator.real_domain()
        
        assert len(domain) == dim
        assert all(b == bounds for b in domain)
    
    def test_default_bin_length_is_zero(self):
        """Test that default bin_length returns 0.
        
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=5)
        assert evaluator.bin_length() == 0
