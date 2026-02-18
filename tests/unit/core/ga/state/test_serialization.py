"""
Unit tests for GaState serialization.

Tests that a GaState instance and all of its data survive round-trips
through Python's pickle module (save / load) and copy.deepcopy().
This covers the "state serialization – save/load functionality" requirement.
"""

import copy
import io
import pickle

import numpy as np
import pytest

from evolvekit.core.Ga.GaState import GaState
from tests.utils.factories.state_factories import configured_state_factory


# ---------------------------------------------------------------------------
# Pickle round-trip (save / load)
# ---------------------------------------------------------------------------

class TestPickleSerialization:
    """Test that GaState can be serialised and deserialised via pickle."""

    def test_empty_state_pickle_round_trip(self):
        """Test that a default GaState survives a pickle round-trip.

        :returns: None
        :raises: None
        """
        state = GaState()

        data = pickle.dumps(state)
        loaded = pickle.loads(data)

        assert isinstance(loaded, GaState)

    def test_configured_state_pickle_round_trip(self):
        """Test that a fully configured GaState survives a pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        data = pickle.dumps(state)
        loaded: GaState = pickle.loads(data)

        assert isinstance(loaded, GaState)

    def test_pickle_preserves_seed(self):
        """Test that seed is preserved after pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.seed == state.seed

    def test_pickle_preserves_probabilities(self):
        """Test that crossover_prob and mutation_prob survive pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.crossover_prob == pytest.approx(state.crossover_prob)
        assert loaded.mutation_prob == pytest.approx(state.mutation_prob)

    def test_pickle_preserves_generation_parameters(self):
        """Test that max_generations, population_size, elite_size survive pickle.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.max_generations == state.max_generations
        assert loaded.population_size == state.population_size
        assert loaded.elite_size == state.elite_size

    def test_pickle_preserves_clamp_strategy(self):
        """Test that real_clamp_strategy survives pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.real_clamp_strategy == state.real_clamp_strategy

    def test_pickle_preserves_population_size(self):
        """Test that the number of individuals in current_population is preserved.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert len(loaded.current_population) == len(state.current_population)

    def test_pickle_preserves_individual_values(self):
        """Test that each individual's fitness value is preserved through pickle.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        original_values = [ind.value for ind in state.current_population]
        loaded_values = [ind.value for ind in loaded.current_population]

        assert loaded_values == pytest.approx(original_values)

    def test_pickle_preserves_individual_chromosomes(self):
        """Test that real chromosomes of individuals survive pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        for orig, restored in zip(state.current_population, loaded.current_population):
            np.testing.assert_array_almost_equal(orig.real_chrom, restored.real_chrom)

    def test_pickle_preserves_generation_counter(self):
        """Test that the statistic_engine generation counter survives pickle.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.statistic_engine.generation == state.statistic_engine.generation

    def test_pickle_preserves_best_individual(self):
        """Test that statistic_engine.best_indiv survives pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        original_best_value = state.statistic_engine.best_indiv.value

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert loaded.statistic_engine.best_indiv.value == pytest.approx(original_best_value)

    def test_pickle_to_file_like_object(self):
        """Test pickle serialisation using a file-like buffer (BytesIO).

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        buffer = io.BytesIO()

        pickle.dump(state, buffer)
        buffer.seek(0)
        loaded: GaState = pickle.load(buffer)

        assert loaded.max_generations == state.max_generations
        assert loaded.seed == state.seed

    def test_loaded_state_is_independent_from_original(self):
        """Test that mutating the loaded state does not affect the original.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        loaded: GaState = pickle.loads(pickle.dumps(state))

        loaded.max_generations = 9999
        loaded.current_population.clear()

        assert state.max_generations == 300
        assert len(state.current_population) == 5


# ---------------------------------------------------------------------------
# deep copy
# ---------------------------------------------------------------------------

class TestDeepCopySerialization:
    """Test that GaState can be duplicated via copy.deepcopy()."""

    def test_deepcopy_returns_new_instance(self):
        """Test that deepcopy returns a distinct GaState object.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        assert cloned is not state

    def test_deepcopy_preserves_scalar_parameters(self):
        """Test that scalar parameters are equal in the cloned state.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        assert cloned.seed == state.seed
        assert cloned.crossover_prob == pytest.approx(state.crossover_prob)
        assert cloned.mutation_prob == pytest.approx(state.mutation_prob)
        assert cloned.max_generations == state.max_generations
        assert cloned.population_size == state.population_size
        assert cloned.elite_size == state.elite_size
        assert cloned.real_clamp_strategy == state.real_clamp_strategy

    def test_deepcopy_population_is_independent(self):
        """Test that modifying the cloned population does not affect the original.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        cloned.current_population.clear()

        assert len(state.current_population) == 5

    def test_deepcopy_individual_chromosomes_are_independent(self):
        """Test that chromosome arrays in cloned individuals are independent copies.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        # Mutate the cloned chromosome
        cloned.current_population[0].real_chrom[0] = 999.0

        assert state.current_population[0].real_chrom[0] != 999.0

    def test_deepcopy_statistic_engine_is_independent(self):
        """Test that the cloned statistic_engine is a separate object.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        assert cloned.statistic_engine is not state.statistic_engine

    def test_deepcopy_preserves_generation_counter(self):
        """Test that the generation counter value is preserved in the clone.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        assert cloned.statistic_engine.generation == state.statistic_engine.generation

    def test_deepcopy_preserves_best_individual_value(self):
        """Test that best_indiv value is preserved in the deep-copied state.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        original_best = state.statistic_engine.best_indiv.value
        cloned = copy.deepcopy(state)

        assert cloned.statistic_engine.best_indiv.value == pytest.approx(original_best)

    def test_deepcopy_of_empty_state(self):
        """Test that deep-copying a default (empty) GaState works without errors.

        :returns: None
        :raises: None
        """
        state = GaState()
        cloned = copy.deepcopy(state)

        assert cloned.current_population == []
        assert cloned.evaluator is None
        assert cloned.crossover_prob == 0
