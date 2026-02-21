"""
Unit tests for GaIsland termination conditions.

Tests that the simulation ends correctly when the maximum generation limit
is reached and when an inspector signals early termination.
"""

from tests.utils.factories.island_factories import minimal_island_factory
from tests.utils.mocks.mock_objects import TerminatingInspector


class TestGaIslandTermination:
    """Test that the simulation terminates under the expected conditions."""

    def test_run_terminates_after_max_generations(self):
        """Test that the simulation stops after exceeding max_generations.

        The loop advances the generation counter past max_generations before
        breaking, so total_generations equals max_generations + 1.

        :returns: None
        :raises: None
        """
        max_gen = 3
        island = minimal_island_factory(max_generations=max_gen)
        results = island.run()

        assert results.total_generations == max_gen + 1

    def test_run_terminates_early_when_inspector_signals_terminate(self):
        """Test that the simulation ends before reaching max_generations when
        the inspector returns GaAction.TERMINATE.

        :returns: None
        :raises: None
        """
        max_gen = 50
        island = minimal_island_factory(max_generations=max_gen)
        island.set_inspector(TerminatingInspector())
        results = island.run()

        assert results.total_generations < max_gen
