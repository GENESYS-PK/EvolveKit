"""
Unit tests for GaIsland population generation.

Tests that run() builds a random initial population of the expected size,
with correctly dimensioned chromosomes, and that every individual has been
evaluated before the first evolution step takes place.
"""

from tests.utils.factories.island_factories import minimal_island_factory


class TestGaIslandPopulationGeneration:
    """Test that a random population is generated correctly on island startup."""

    def test_population_has_correct_size(self):
        """Test that current_population contains exactly population_size individuals.

        :returns: None
        :raises: None
        """
        population_size = 20
        island = minimal_island_factory(population_size=population_size)
        island.run()

        assert len(island.current_population) == population_size

    def test_individual_real_chromosomes_have_correct_dimension(self):
        """Test that each generated individual has a real chromosome of the expected length.

        :returns: None
        :raises: None
        """
        dim = 15
        island = minimal_island_factory(dim=dim)
        island.run()

        for individual in island.current_population:
            assert len(individual.real_chrom) == dim

    def test_each_individual_has_been_evaluated(self):
        """Test that every individual in the population carries a float fitness value.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        island.run()

        for individual in island.current_population:
            assert isinstance(individual.value, float)
