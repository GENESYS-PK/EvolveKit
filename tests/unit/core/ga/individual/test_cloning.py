"""
Unit tests for GaIndividual cloning.

Tests deep copy behavior to ensure independent copies are created.
"""

import pytest
import numpy as np
from copy import deepcopy

from evolvekit.core.Ga.GaIndividual import GaIndividual


class TestIndividualCloning:
    """Test deep copy behavior of individuals."""
    
    def test_deepcopy_creates_independent_copy(self):
        """Test that deepcopy creates an independent individual.
        
        :returns: None
        :raises: None
        """
        original = GaIndividual(
            real_chrom=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            value=10.0
        )
        clone = deepcopy(original)
        
        assert clone is not original
        assert np.array_equal(clone.real_chrom, original.real_chrom)
        assert np.array_equal(clone.bin_chrom, original.bin_chrom)
        assert clone.value == pytest.approx(original.value)
    
    def test_deepcopy_real_chromosome_independence(self):
        """Test that cloned real chromosome is independent.
        
        :returns: None
        :raises: None
        """
        original = GaIndividual(real_chrom=np.array([1.0, 2.0, 3.0], dtype=np.float64))
        clone = deepcopy(original)
        
        clone.real_chrom[0] = 999.0
        
        assert original.real_chrom[0] == pytest.approx(1.0)
        assert clone.real_chrom[0] == pytest.approx(999.0)
    
    def test_deepcopy_binary_chromosome_independence(self):
        """Test that cloned binary chromosome is independent.
        
        :returns: None
        :raises: None
        """
        bin_chrom = np.array([255, 128, 64], dtype=np.uint8)
        original = GaIndividual(bin_chrom=bin_chrom)
        clone = deepcopy(original)
        
        clone.bin_chrom[0] = 0
        
        assert original.bin_chrom[0] == 255
        assert clone.bin_chrom[0] == 0
    
    def test_deepcopy_fitness_independence(self):
        """Test that fitness values are independent after cloning.
        
        :returns: None
        :raises: None
        """
        original = GaIndividual(value=50.0)
        clone = deepcopy(original)
        
        clone.value = 100.0
        
        assert original.value == pytest.approx(50.0)
        assert clone.value == pytest.approx(100.0)
    
    def test_deepcopy_empty_individual(self):
        """Test cloning an empty individual.
        
        :returns: None
        :raises: None
        """
        original = GaIndividual()
        clone = deepcopy(original)
        
        assert clone is not original
        assert len(clone.real_chrom) == 0
        assert len(clone.bin_chrom) == 0
