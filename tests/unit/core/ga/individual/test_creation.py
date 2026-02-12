"""
Unit tests for GaIndividual creation.

Tests proper initialization with real, binary, and mixed chromosomes.
"""

import pytest
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual


class TestIndividualCreation:
    """Test individual creation with different chromosome types."""
    
    def test_empty_individual_creation(self):
        """Test creating individual with no chromosomes.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual()
        
        assert len(individual.real_chrom) == 0
        assert len(individual.bin_chrom) == 0
        assert individual.value == pytest.approx(0.0)
    
    def test_real_chromosome_creation(self):
        """Test creating individual with real chromosome.
        
        :returns: None
        :raises: None
        """
        real_chrom = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        individual = GaIndividual(real_chrom=real_chrom)
        
        assert len(individual.real_chrom) == 3
        assert np.array_equal(individual.real_chrom, real_chrom)
        assert len(individual.bin_chrom) == 0
        assert individual.value == pytest.approx(0.0)
    
    def test_binary_chromosome_creation(self):
        """Test creating individual with binary chromosome.
        
        :returns: None
        :raises: None
        """
        bin_chrom = np.packbits(np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8))
        individual = GaIndividual(bin_chrom=bin_chrom)
        
        assert len(individual.bin_chrom) == 1
        assert np.array_equal(individual.bin_chrom, bin_chrom)
        assert len(individual.real_chrom) == 0
        assert individual.value == pytest.approx(0.0)
    
    def test_mixed_chromosome_creation(self):
        """Test creating individual with both real and binary chromosomes.
        
        :returns: None
        :raises: None
        """
        real_chrom = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        bin_chrom = np.packbits(np.array([1, 1, 1, 1], dtype=np.uint8))
        individual = GaIndividual(real_chrom=real_chrom, bin_chrom=bin_chrom)
        
        assert len(individual.real_chrom) == 3
        assert len(individual.bin_chrom) == 1
        assert np.array_equal(individual.real_chrom, real_chrom)
        assert np.array_equal(individual.bin_chrom, bin_chrom)
        assert individual.value == pytest.approx(0.0)
    
    @pytest.mark.parametrize("size", [1, 100])
    def test_real_chromosome_different_sizes(self, size):
        """Test creating individuals with different chromosome sizes.
        
        :param size: Size of the chromosome
        :returns: None
        :raises: None
        """
        real_chrom = np.ones(size, dtype=np.float64)
        individual = GaIndividual(real_chrom=real_chrom)
        
        assert len(individual.real_chrom) == size
