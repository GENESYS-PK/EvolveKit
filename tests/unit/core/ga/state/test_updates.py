"""
Unit tests for GaState updates.

Tests generation counter increments, best/worst individual tracking,
stagnation detection, and statistical field updates driven by
GaStatisticEngine operating on GaState.
"""

import pytest
import numpy as np

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
# Stagnation counter
# ---------------------------------------------------------------------------

class TestStagnationCounter:
    """Test the stagnation counter that tracks lack of improvement."""

    def test_start_sets_stagnation_to_zero(self):
        """Test that start() initialises stagnation to 0.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0])
        state.statistic_engine.start(state)

        assert state.statistic_engine.stagnation == 0

    def test_stagnation_increments_when_best_unchanged(self):
        """Test that stagnation counter increments when best individual does not change.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0], GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)  # generation 1, stagnation resets to 0 (no previous best)

        # Same population → same best value
        state.statistic_engine.advance(state)

        assert state.statistic_engine.stagnation == 1

    def test_stagnation_resets_when_best_improves(self):
        """Test that stagnation counter resets to 0 when best individual improves.

        :returns: None
        :raises: None
        """
        state = state_with_population_factory([1.0, 2.0, 3.0], GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)
        state.statistic_engine.advance(state)  # stagnation == 1

        # Improve population
        state.current_population = population_with_values_factory([0.1, 0.2, 0.3])
        state.statistic_engine.advance(state)

        assert state.statistic_engine.stagnation == 0

    def test_stagnation_accumulates_over_multiple_unchanged_generations(self):
        """Test that stagnation accumulates correctly across many unchanged generations.

        :returns: None
        :raises: None
        """
        n_stagnant = 4
        state = state_with_population_factory([1.0, 2.0, 3.0], GaExtremum.MINIMUM)
        state.statistic_engine.start(state)
        state.statistic_engine.advance(state)  # first advance, no prev_best yet → stagnation stays 0

        for _ in range(n_stagnant):
            state.statistic_engine.advance(state)

        assert state.statistic_engine.stagnation == n_stagnant


# ---------------------------------------------------------------------------
# Statistical field updates (mean, median, stdev)
# ---------------------------------------------------------------------------

class TestStatisticalFieldUpdates:
    """Test that mean, median, and standard deviation are computed correctly."""

    def test_refresh_computes_correct_mean(self):
        """Test that refresh() sets mean to the population average.

        :returns: None
        :raises: None
        """
        values = [1.0, 3.0, 5.0]
        state = state_with_population_factory(values)
        state.statistic_engine.start(state)
        state.statistic_engine.refresh(state)

        assert state.statistic_engine.mean == pytest.approx(np.mean(values))

    def test_refresh_computes_correct_median(self):
        """Test that refresh() sets median correctly.

        :returns: None
        :raises: None
        """
        values = [1.0, 2.0, 10.0]
        state = state_with_population_factory(values)
        state.statistic_engine.start(state)
        state.statistic_engine.refresh(state)

        assert state.statistic_engine.median == pytest.approx(np.median(values))

    def test_refresh_computes_correct_stdev(self):
        """Test that refresh() sets standard deviation correctly.

        :returns: None
        :raises: None
        """
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        state = state_with_population_factory(values)
        state.statistic_engine.start(state)
        state.statistic_engine.refresh(state)

        assert state.statistic_engine.stdev == pytest.approx(np.std(values))

    def test_advance_computes_statistics_same_as_refresh(self):
        """Test that advance() produces the same stats as refresh() for a given population.

        :returns: None
        :raises: None
        """
        values = [1.0, 4.0, 7.0]
        state_adv = state_with_population_factory(values)
        state_ref = state_with_population_factory(values)

        state_adv.statistic_engine.start(state_adv)
        state_adv.statistic_engine.advance(state_adv)

        state_ref.statistic_engine.start(state_ref)
        state_ref.statistic_engine.refresh(state_ref)

        assert state_adv.statistic_engine.mean == pytest.approx(state_ref.statistic_engine.mean)
        assert state_adv.statistic_engine.median == pytest.approx(state_ref.statistic_engine.median)
        assert state_adv.statistic_engine.stdev == pytest.approx(state_ref.statistic_engine.stdev)

    def test_uniform_population_has_zero_stdev(self):
        """Test that a population where all individuals share the same value has stdev == 0.

        :returns: None
        :raises: None
        """
        values = [5.0] * 6
        state = state_with_population_factory(values)
        state.statistic_engine.start(state)
        state.statistic_engine.refresh(state)

        assert state.statistic_engine.stdev == pytest.approx(0.0)


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
