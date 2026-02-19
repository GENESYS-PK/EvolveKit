"""
Unit tests for GaState serialization.

Representative smoke tests verifying that a GaState instance survives a
pickle round-trip (save / load) and copy.deepcopy() without data loss.
"""

import copy
import pickle

import pytest

from evolvekit.core.Ga.GaState import GaState
from tests.utils.factories.state_factories import configured_state_factory


class TestPickleSerialization:
    """Smoke test – GaState can be serialised and deserialised via pickle."""

    def test_configured_state_pickle_round_trip(self):
        """Test that a fully configured GaState survives a pickle round-trip.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()

        loaded: GaState = pickle.loads(pickle.dumps(state))

        assert isinstance(loaded, GaState)
        assert loaded.seed == state.seed
        assert loaded.crossover_prob == pytest.approx(state.crossover_prob)
        assert loaded.max_generations == state.max_generations
        assert len(loaded.current_population) == len(state.current_population)


class TestDeepCopySerialization:
    """Smoke test – GaState can be duplicated via copy.deepcopy()."""

    def test_deepcopy_returns_independent_instance(self):
        """Test that deepcopy returns a distinct, independent GaState object.

        :returns: None
        :raises: None
        """
        state = configured_state_factory()
        cloned = copy.deepcopy(state)

        assert cloned is not state
        assert cloned.statistic_engine is not state.statistic_engine

        cloned.current_population.clear()
        assert len(state.current_population) == 5
