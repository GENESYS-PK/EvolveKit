"""
Unit tests for GaIndividual fitness assignment.

Tests setting and getting fitness values.
"""

import pytest
import numpy as np

from evolvekit.core.Ga.GaIndividual import GaIndividual


class TestFitnessAssignment:
    """Test fitness value assignment and retrieval."""
    
    def test_default_fitness_is_zero(self):
        """Test that default fitness value is 0.0.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual()
        
        assert individual.value == pytest.approx(0.0)
    
    def test_fitness_assignment_at_creation(self):
        """Test setting fitness at creation time.
        
        :returns: None
        :raises: None
        """
        fitness = 123.456
        individual = GaIndividual(value=fitness)
        
        assert individual.value == pytest.approx(fitness)
    
    def test_fitness_assignment_after_creation(self):
        """Test modifying fitness after creation.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual()
        individual.value = 99.9
        
        assert individual.value == pytest.approx(99.9)
    
    def test_negative_fitness(self):
        """Test that negative fitness values work.
        
        :returns: None
        :raises: None
        """
        individual = GaIndividual(value=-50.5)
        
        assert individual.value == pytest.approx(-50.5)
    
    def test_very_large_fitness(self):
        """Test very large fitness values.
        
        :returns: None
        :raises: None
        """
        large_fitness = 1e12
        individual = GaIndividual(value=large_fitness)
        
        assert individual.value == pytest.approx(large_fitness)
