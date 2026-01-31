"""
Unit tests for GaEvaluator with different chromosome types.

Tests evaluator behavior with binary, real-valued, and mixed
chromosomes to ensure proper handling of different representations.
"""
import pytest
import numpy as np
from typing import List, Tuple

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from tests.utils.factories.individual_factories import (
    random_individual_factory,
    create_individual
)


class BinaryEvaluator(GaEvaluator):
    """Evaluator that uses binary chromosome - counts 1s in bit string."""
    
    def __init__(self, bin_len: int = 10):
        self._bin_len = bin_len
    
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        return float(np.sum(args.bin_chrom[:self._bin_len]))
    
    def extremum(self) -> GaExtremum:
        return GaExtremum.MAXIMUM
    
    def bin_length(self) -> int:
        return self._bin_len


class MixedChromosomeEvaluator(GaEvaluator):
    """Evaluator that uses both real and binary chromosomes."""
    
    def __init__(self, real_dim: int = 3, bin_len: int = 5):
        self._real_dim = real_dim
        self._bin_len = bin_len
    
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        real_part = np.sum(np.square(args.real_chrom[:self._real_dim]))
        bin_part = np.sum(args.bin_chrom[:self._bin_len])
        return float(real_part + bin_part)
    
    def extremum(self) -> GaExtremum:
        return GaExtremum.MINIMUM
    
    def real_domain(self) -> List[Tuple[float, float]]:
        return [(-5.0, 5.0)] * self._real_dim
    
    def bin_length(self) -> int:
        return self._bin_len


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
        evaluator = BinaryEvaluator(bin_len=bin_len)
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
        evaluator = MixedChromosomeEvaluator(real_dim=real_dim, bin_len=bin_len)
        
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
        evaluator = MixedChromosomeEvaluator(real_dim=real_dim, bin_len=bin_len)
        
        assert len(evaluator.real_domain()) == real_dim
        assert evaluator.bin_length() == bin_len
