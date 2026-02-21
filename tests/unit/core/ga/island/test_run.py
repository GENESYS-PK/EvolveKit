"""
Unit tests for GaIsland run behavior.

Tests population generation after run(), termination conditions
(max generations and inspector-driven), and the GaResults statistics
returned after a completed simulation.
"""

import pytest

from evolvekit.core.Ga.GaResults import GaResults
from tests.utils.factories.island_factories import minimal_island_factory
from tests.utils.mocks.mock_objects import TerminatingInspector


# ---------------------------------------------------------------------------
# Population generation
# ---------------------------------------------------------------------------

class TestGaIslandPopulationGeneration:
    """Test that run() generates and evaluates a population of the expected size."""

    def test_population_has_correct_size_after_run(self):
        """Test that current_population contains exactly population_size individuals.

        :returns: None
        :raises: None
        """
        population_size = 20
        island = minimal_island_factory(population_size=population_size)
        island.run()

        assert len(island.current_population) == population_size

    def test_each_individual_has_a_fitness_value_after_run(self):
        """Test that every individual in the population has been evaluated.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        island.run()

        for individual in island.current_population:
            assert isinstance(individual.value, float)


# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------

class TestGaIslandTermination:
    """Test that the simulation terminates under the expected conditions."""

    def test_run_returns_ga_results_instance(self):
        """Test that run() always returns a GaResults object.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory(max_generations=1)
        results = island.run()

        assert isinstance(results, GaResults)

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

    def test_run_terminates_when_inspector_signals_terminate(self):
        """Test that the simulation ends early when the inspector returns TERMINATE.

        :returns: None
        :raises: None
        """
        max_gen = 50
        island = minimal_island_factory(max_generations=max_gen)
        island.set_inspector(TerminatingInspector())
        results = island.run()

        assert results.total_generations < max_gen


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestGaIslandStatistics:
    """Test that GaResults contains valid statistics after a completed run."""

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
