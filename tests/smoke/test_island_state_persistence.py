"""
Smoke tests for GaIsland state persistence.

These tests exercise multiple components together (island run, pickle, deepcopy)
and are suited for CI pipeline smoke checks rather than unit test isolation.
"""

import copy
import pickle

from evolvekit.core.Ga.GaIsland import GaIsland
from tests.utils.factories.island_factories import minimal_island_factory


class TestPickleSerialization:
    """Smoke test – GaIsland can be serialised and deserialised via pickle."""

    def test_island_pickle_round_trip(self):
        """Test that a GaIsland instance with a completed run survives a pickle round-trip.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory(max_generations=1)
        island.run()

        loaded: GaIsland = pickle.loads(pickle.dumps(island))

        assert isinstance(loaded, GaIsland)
        assert loaded.max_generations == island.max_generations
        assert loaded.population_size == island.population_size


class TestDeepCopySerialization:
    """Smoke test – GaIsland can be duplicated via copy.deepcopy()."""

    def test_deepcopy_returns_independent_instance(self):
        """Test that deepcopy produces a distinct, independent GaIsland object.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        cloned = copy.deepcopy(island)

        assert cloned is not island
        assert cloned.statistic_engine is not island.statistic_engine

        cloned.population_size = 9999
        assert island.population_size == 10  # original unchanged
