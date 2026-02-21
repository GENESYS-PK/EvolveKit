"""
Unit tests for GaIsland evolution step.

Tests the elitism code path that runs during single generation execution.
"""

from evolvekit.core.Ga.GaResults import GaResults
from tests.utils.factories.island_factories import minimal_island_factory


class TestGaIslandEvolutionStep:
    """Test structural correctness of a single generation execution."""

    def test_run_with_elitism_completes_without_error(self):
        """Test that the elitism code path executes without raising exceptions.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory(max_generations=3)
        island.set_elite_count(2)
        results = island.run()

        assert isinstance(results, GaResults)
