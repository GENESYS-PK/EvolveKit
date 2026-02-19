"""
Unit tests for GaState updates.

Tests that GaState correctly reflects changes to its generation counter,
best/worst individual tracking and population fields after operations.

Note: Tests covering GaStatisticEngine internals (stagnation logic,
mean/median/stdev calculations) belong in the GaStatisticEngine test suite.
"""

import pytest

from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from tests.utils.factories.population_factories import population_factory
from tests.utils.factories.state_factories import (
    population_with_values_factory,
    state_with_population_factory,
)
from tests.utils.fixtures.test_utilities import find_best_individual


# ---------------------------------------------------------------------------
# Generation counter
# ---------------------------------------------------------------------------

class TestGenerationCounter:
    """Test that the generation counter is incremented correctly."""

    def test_start_sets_generation_to_zero(self):
        """Test that start() initialises generation to 0.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0])
        state.statistic_engine.start(state)

        assert state.statistic_engine.generation == 0

    def test_advance_increments_generation_by_one(self):
        """Test that a single advance() call increments generation from 0 to 1.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0])
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        assert state.statistic_engine.generation == 1

    def test_advance_multiple_times_increments_correctly(self):
        """Test that generation is incremented by one for each advance() call.

        :returns: None
        :raises: None
        """
        n_advances = 5
        state = state_with_population_factory([1.0, 2.0, 3.0])
        state.statistic_engine.start(state)

        for _ in range(n_advances):
            state.statistic_engine.advance(state)

        assert state.statistic_engine.generation == n_advances

    def test_refresh_does_not_change_generation(self):
        """Test that refresh() does NOT increment the generation counter.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0])
        state.statistic_engine.start(state)
        state.statistic_engine.refresh(state)

        assert state.statistic_engine.generation == 0


# ---------------------------------------------------------------------------
# Best fitness tracking – MINIMUM evaluator
# ---------------------------------------------------------------------------

class TestBestFitnessTrackingMinimum:
    """Test best individual tracking when minimising fitness."""

    def test_start_sets_best_indiv_to_none(self):
        """Test that start() initialises best_indiv to None.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([3.0, 1.0, 2.0])
        state.statistic_engine.start(state)

        assert state.statistic_engine.best_indiv is None

    def test_advance_identifies_minimum_as_best(self):
        """Test that the individual with the lowest value becomes best_indiv.

        Cross-verified with find_best_individual utility.

        :returns: None
        :raises: None
        """
        values = [3.0, 1.0, 2.0]
        state = state_with_population_factory(values, GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        _, expected_best, _ = find_best_individual(
            state.current_population, values, minimize=True
        )
        assert state.statistic_engine.best_indiv.value == pytest.approx(expected_best)

    def test_advance_identifies_maximum_as_worst(self):
        """Test that the individual with the highest value becomes worst_indiv when minimising.

        Cross-verified with find_best_individual utility (worst = best when maximising).

        :returns: None
        :raises: None
        """
        values = [3.0, 1.0, 2.0]
        state = state_with_population_factory(values, GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        _, expected_worst, _ = find_best_individual(
            state.current_population, values, minimize=False
        )
        assert state.statistic_engine.worst_indiv.value == pytest.approx(expected_worst)

    def test_best_indiv_updates_when_population_improves(self):
        """Test that best_indiv is updated after population changes.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([3.0, 2.0, 1.0], GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        assert state.statistic_engine.best_indiv.value == pytest.approx(1.0)

        # Simulate improved population
        state.current_population = population_with_values_factory([0.5, 0.3, 0.1])
        state.statistic_engine.advance(state)

        assert state.statistic_engine.best_indiv.value == pytest.approx(0.1)

    def test_best_indiv_is_deep_copy_of_individual(self):
        """Test that best_indiv is a deep copy, not a reference to the original.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([3.0, 1.0, 2.0], GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        original_best = state.current_population[1]  # value == 1.0
        captured = state.statistic_engine.best_indiv

        # Mutate the original
        original_best.value = 99.0

        assert captured.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Best fitness tracking – MAXIMUM evaluator
# ---------------------------------------------------------------------------

class TestBestFitnessTrackingMaximum:
    """Test best individual tracking when maximising fitness."""

    def test_advance_identifies_maximum_as_best(self):
        """Test that the individual with the highest value becomes best_indiv.

        Cross-verified with find_best_individual utility.

        :returns: None
        :raises: None
        """
        values = [3.0, 1.0, 2.0]
        state = state_with_population_factory(values, GaExtremum.MAXIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        _, expected_best, _ = find_best_individual(
            state.current_population, values, minimize=False
        )
        assert state.statistic_engine.best_indiv.value == pytest.approx(expected_best)

    def test_advance_identifies_minimum_as_worst(self):
        """Test that the individual with the lowest value becomes worst_indiv when maximising.

        Cross-verified with find_best_individual utility (worst = best when minimising).

        :returns: None
        :raises: None
        """
        values = [3.0, 1.0, 2.0]
        state = state_with_population_factory(values, GaExtremum.MAXIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)

        _, expected_worst, _ = find_best_individual(
            state.current_population, values, minimize=True
        )
        assert state.statistic_engine.worst_indiv.value == pytest.approx(expected_worst)


# ---------------------------------------------------------------------------
# Population assignment
# ---------------------------------------------------------------------------

class TestPopulationAssignment:
    """Test assigning populations to state fields."""

    def test_assign_current_population(self):
        """Test that current_population can be assigned and retrieved.

        :returns: None
        :raises: None
        """
        state = GaState()
        pop = population_factory(dimension=3, population_size=10, seed=0)
        state.current_population = pop

        assert len(state.current_population) == 10

    def test_assign_selected_population(self):
        """Test that selected_population can be assigned and retrieved.

        :returns: None
        :raises: None
        """
        state = GaState()
        pop = population_factory(dimension=3, population_size=5, seed=1)
        state.selected_population = pop

        assert len(state.selected_population) == 5

    def test_assign_offspring_population(self):
        """Test that offspring_population can be assigned and retrieved.

        :returns: None
        :raises: None
        """
        state = GaState()
        pop = population_factory(dimension=3, population_size=8, seed=2)
        state.offspring_population = pop

        assert len(state.offspring_population) == 8

    def test_assign_elite_population(self):
        """Test that elite_population can be assigned and retrieved.

        :returns: None
        :raises: None
        """
        state = GaState()
        pop = population_factory(dimension=3, population_size=3, seed=3)
        state.elite_population = pop

        assert len(state.elite_population) == 3
