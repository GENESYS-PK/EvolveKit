"""
Unit tests for GaEvaluator with different chromosome types.

Tests evaluator behavior with binary, real-valued, and mixed
chromosomes to ensure proper handling of different representations.
"""
import pytest
import numpy as np
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.factories.individual_factories import (
    random_individual_factory,
    create_individual
)
from tests.utils.mocks.mock_objects import (
    MockBinaryEvaluator,
    MockMixedEvaluator
)


class TestDifferentChromosomeTypes:
    """Test evaluator with binary vs real-valued chromosomes."""
    
    @pytest.mark.parametrize("bin_values,expected", [
        ([1, 0, 1, 1, 0, 1, 0, 1], 5.0),
        ([0, 0, 0, 0, 0, 0, 0, 0], 0.0),
        ([1, 1, 1, 1, 1, 1, 1, 1], 8.0),
    ])
    def test_binary_chromosome(self, bin_values, expected):
        """Test evaluator with binary chromosome.
        
        :param bin_values: Binary chromosome values
        :param expected: Expected fitness value
        :returns: None
        :raises: None
        """
        bin_len = len(bin_values)
        evaluator = MockBinaryEvaluator(bin_len=bin_len)
        individual = GaIndividual(bin_chrom=np.array(bin_values, dtype=np.uint8))
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        assert fitness == expected
    
    def test_mixed_chromosome_types(self):
        """Test evaluator using both real and binary chromosomes.
        
        Real part: 1^2 + 2^2 = 5, bin part: 1+1+1 = 3, total = 8.
        
        :returns: None
        :raises: None
        """
        real_dim = 2
        bin_len = 3
        evaluator = MockMixedEvaluator(real_dim=real_dim, bin_len=bin_len)
        
        individual = GaIndividual(
            real_chrom=np.array([1.0, 2.0], dtype=np.float64),
            bin_chrom=np.array([1, 1, 1], dtype=np.uint8)
        )
        args = GaEvaluatorArgs(individual)
        
        fitness = evaluator.evaluate(args)
        assert fitness == 8.0
    
    def test_mixed_evaluator_domain_and_bin_length(self):
        """Test mixed evaluator returns both domain and bin_length.
        
        :returns: None
        :raises: None
        """
        real_dim = 4
        bin_len = 6
        evaluator = MockMixedEvaluator(real_dim=real_dim, bin_len=bin_len)
        
        assert len(evaluator.real_domain()) == real_dim
        assert evaluator.bin_length() == bin_len
