"""
Unit tests for GaEvaluator error handling.

Tests proper error handling for malformed inputs, None values,
and edge cases like NaN and infinity.
"""

import pytest
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.factories.individual_factories import create_individual


class TestInvalidInputHandling:
    """Test proper error handling for malformed inputs."""
    
    def test_none_args_raises_error(self):
        """Test that None as args raises AttributeError.
        
        :returns: None
        :raises AttributeError: When args is None
        """
        evaluator = SphereEvaluator(dim=5)
        
        with pytest.raises(AttributeError):
            evaluator.evaluate(None)
    
    @pytest.mark.parametrize("invalid_arg", ["string", 42, [1, 2, 3]])
    def test_wrong_type_args(self, invalid_arg):
        """Test that wrong type for args raises error.
        
        :param invalid_arg: Invalid argument of wrong type
        :returns: None
        :raises AttributeError: When args is not GaEvaluatorArgs
        """
        evaluator = SphereEvaluator(dim=5)
        with pytest.raises(AttributeError):
            evaluator.evaluate(invalid_arg)
    
    def test_insufficient_chromosome_length(self):
        """Test evaluator handles insufficient chromosome length gracefully.
        
        :returns: None
        :raises: None
        """
        dim = 5
        evaluator = SphereEvaluator(dim=dim)
        individual = create_individual([1.0, 2.0])
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        assert isinstance(fitness, float)
    
    @pytest.mark.parametrize("values,check_fn", [
        ([1.0, np.nan, 3.0], lambda x: np.isnan(x)),
        ([1.0, np.inf, 3.0], lambda x: np.isinf(x)),
        ([1.0, -np.inf, 3.0], lambda x: np.isinf(x)),
    ])
    def test_special_float_values(self, values, check_fn):
        """Test evaluator behavior with NaN and infinity.
        
        :param values: Chromosome values containing special floats
        :param check_fn: Function to check expected result
        :returns: None
        :raises: None
        """
        evaluator = SphereEvaluator(dim=3)
        individual = create_individual(values)
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        assert check_fn(fitness)
