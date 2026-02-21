"""
Unit tests for GaIsland statistics collection.

Tests that GaResults carries valid performance metrics after a completed
simulation: elapsed time, best individual fitness, and generation count.
"""

from tests.utils.factories.island_factories import minimal_island_factory


class TestGaIslandStatisticsCollection:
    """Test that performance metrics are correctly tracked and returned."""

    def test_results_total_time_is_non_negative(self):
        """Test that the elapsed CPU time recorded in results is non-negative.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory(max_generations=1)
        results = island.run()

        assert results.total_time >= 0

    def test_results_best_value_is_float(self):
        """Test that the best individual's fitness value is a float.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory(max_generations=1)
        results = island.run()

        assert isinstance(results.value, float)
