"""
Unit tests for GaStatisticEngine – statistics aggregation across multiple
generations.

Covers: generation counter increments, stagnation counter behaviour (reset
on improvement / accumulate on plateau), and timing fields updated after
each advance.
"""

import numpy as np
import pytest

from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from tests.utils.factories.state_factories import (
    statistic_engine_factory,
    population_with_values_factory,
)


class TestGenerationCounter:
    """Tests that verify the generation counter is managed correctly."""

    def test_generation_starts_at_zero_after_start(self):
        """After calling start() the generation counter must equal zero."""
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        assert engine.generation == 0

    @pytest.mark.parametrize("n_advances", [1, 5, 10])
    def test_generation_equals_number_of_advances(self, n_advances):
        """Generation counter must always equal the total number of advance() calls.

        :param n_advances: How many times to call advance().
        """
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        for _ in range(n_advances):
            engine.advance(state)

        assert engine.generation == n_advances

    def test_generation_does_not_change_on_refresh(self):
        """refresh() must update statistics but must NOT change the generation counter."""
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        engine.advance(state)
        gen_before = engine.generation

        engine.refresh(state)

        assert engine.generation == gen_before


class TestStagnationCounter:
    """Tests for the stagnation counter: increment on plateau, reset on improvement."""

    def test_stagnation_starts_at_zero_after_start(self):
        """After start() the stagnation counter must be zero."""
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        assert engine.stagnation == 0

    @pytest.mark.parametrize("n_stagnant", [1, 4])
    def test_stagnation_accumulates_over_unchanged_generations(self, n_stagnant):
        """Stagnation must accumulate correctly across consecutive generations where
        the best fitness is unchanged.

        :param n_stagnant: Number of stagnant advances after the first generation.
        """
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        engine.advance(state)  # gen 1 – no previous best → stagnation = 0
        for _ in range(n_stagnant):
            engine.advance(state)  # best stays at 1.0 every time

        assert engine.stagnation == n_stagnant

    def test_stagnation_resets_to_zero_when_best_fitness_improves(self):
        """When the best individual's fitness improves, stagnation must reset to 0."""
        engine, state = statistic_engine_factory([5.0, 6.0, 7.0])
        engine.advance(state)  # gen 1, stagnation = 0
        engine.advance(state)  # gen 2, best unchanged → stagnation = 1

        state.current_population = population_with_values_factory([1.0, 2.0, 3.0])
        engine.advance(state)  # gen 3, best improved → stagnation resets to 0

        assert engine.stagnation == 0

    def test_stagnation_resumes_accumulating_after_reset(self):
        """After a reset, stagnation must start accumulating again when the best
        fitness remains unchanged.
        """
        engine, state = statistic_engine_factory([5.0, 6.0, 7.0])
        engine.advance(state)  # gen 1 – stagnation 0

        state.current_population = population_with_values_factory([1.0, 2.0, 3.0])
        engine.advance(state)  # gen 2 – improved → stagnation 0
        engine.advance(state)  # gen 3 – stagnant → stagnation 1

        assert engine.stagnation == 1


class TestTimingAndReset:
    """Tests for timing fields and full engine reset via start()."""

    def test_start_time_is_non_negative_after_start(self):
        """start_time must be non-negative after start() is called."""
        engine, state = statistic_engine_factory([1.0, 2.0])
        assert engine.start_time >= 0.0

    def test_last_time_gte_start_time_after_advance(self):
        """last_time must be ≥ start_time after advance() (process time is
        monotonically non-decreasing).
        """
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        engine.advance(state)

        assert engine.last_time >= engine.start_time

    def test_start_resets_generation_stagnation_and_statistics(self):
        """Calling start() on a used engine must reset generation, stagnation,
        and clear best/worst individuals.
        """
        engine, state = statistic_engine_factory([1.0, 2.0, 3.0])
        for _ in range(3):
            engine.advance(state)

        engine.start(state)

        assert engine.generation == 0
        assert engine.stagnation == 0
        assert engine.best_indiv is None
        assert engine.worst_indiv is None


class TestStatisticsReflectCurrentPopulation:
    """Tests that statistics always reflect the most recent population."""

    def test_statistics_updated_after_population_replacement(self):
        """mean and best_indiv must reflect the new population after replacing it
        between two advance() calls.
        """
        engine, state = statistic_engine_factory(
            [10.0, 20.0, 30.0], extremum=GaExtremum.MINIMUM
        )
        engine.advance(state)

        new_values = [1.0, 2.0, 3.0]
        state.current_population = population_with_values_factory(new_values)
        engine.advance(state)

        assert engine.best_indiv.value == pytest.approx(1.0)
        assert engine.mean == pytest.approx(float(np.mean(new_values)))
