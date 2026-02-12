"""
Unit tests for GaIndividual state changes.

Tests chromosome mutations and state tracking after operations.
"""

import pytest
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual


class TestStateChanges:
    """Test chromosome mutation and state tracking."""
    
    def test_real_chromosome_modification(self):
        """Test that real chromosome can be modified.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual(real_chrom=np.array([1.0, 2.0, 3.0], dtype=np.float64), value=5.0)
        
        individual.real_chrom[0] = 10.0
        
        assert individual.real_chrom[0] == pytest.approx(10.0)
        assert individual.value == pytest.approx(5.0)
    
    def test_binary_chromosome_modification(self):
        """Test that binary chromosome can be modified.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual(bin_chrom=np.array([255, 128], dtype=np.uint8), value=10.0)
        
        individual.bin_chrom[0] = 0
        
        assert individual.bin_chrom[0] == 0
        assert individual.value == pytest.approx(10.0)
    
    def test_chromosome_replacement(self):
        """Test replacing entire chromosome.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual(
            real_chrom=np.array([1.0, 2.0], dtype=np.float64),
            bin_chrom=np.array([255], dtype=np.uint8),
            value=7.5
        )
        new_chrom = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        
        individual.real_chrom = new_chrom
        
        assert len(individual.real_chrom) == 3
        assert np.array_equal(individual.real_chrom, new_chrom)
        assert individual.bin_chrom[0] == 255
        assert individual.value == pytest.approx(7.5)
    
    def test_mixed_chromosome_independent_modifications(self):
        """Test that real and binary chromosomes can be modified independently.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual(
            real_chrom=np.array([1.0, 2.0], dtype=np.float64),
            bin_chrom=np.array([255, 128], dtype=np.uint8)
        )
        
        individual.real_chrom[0] = 10.0
        individual.bin_chrom[0] = 0
        
        assert individual.real_chrom[0] == pytest.approx(10.0)
        assert individual.real_chrom[1] == pytest.approx(2.0)
        assert individual.bin_chrom[0] == 0
        assert individual.bin_chrom[1] == 128
