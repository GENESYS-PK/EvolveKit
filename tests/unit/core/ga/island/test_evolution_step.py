"""
Unit tests for GaIsland evolution step.

Behavioral tests verifying that elitism works correctly:
- running with elitism produces a better (or equal) best fitness than without it
- the best fitness value never deteriorates across consecutive generations
"""

import pytest

from tests.utils.factories.island_factories import minimal_island_factory
from tests.utils.mocks.mock_objects import (
    AfterNGenerationsInspector,
    FitnessCapturingInspector,
)


class TestGaIslandElitismSelection:
    """Behavioral tests for elitism during the evolution step."""

    def test_elitism_produces_better_or_equal_best_fitness_than_no_elitism(self):
        """Test that running with elitism results in a best fitness value that is
        at least as good as running without it, given the same initial seed.

        Both runs start from the same seed (same initial population). After one
        generation, the minimum fitness in the elitism run must be <= the minimum
        fitness in the no-elitism run, because the elite individual is guaranteed
        to survive.

        :returns: None
        :raises: None
        """
        seed = 42
        population_size = 20

        inspector_elite = AfterNGenerationsInspector(n=2)
        island_elite = minimal_island_factory(
            population_size=population_size, max_generations=50
        )
        island_elite.set_elite_count(2)
        island_elite.set_seed(seed)
        island_elite.set_inspector(inspector_elite)
        island_elite.run()
        best_with_elitism = min(ind.value for ind in island_elite.current_population)

        inspector_no_elite = AfterNGenerationsInspector(n=2)
        island_no_elite = minimal_island_factory(
            population_size=population_size, max_generations=50
        )
        island_no_elite.set_elite_count(0)
        island_no_elite.set_seed(seed)
        island_no_elite.set_inspector(inspector_no_elite)
        island_no_elite.run()
        best_without_elitism = min(ind.value for ind in island_no_elite.current_population)

        assert best_with_elitism <= best_without_elitism

    @pytest.mark.parametrize("elite_size", [1, 2, 5])
    def test_best_fitness_never_increases_with_elitism(self, elite_size):
        """Test that the best fitness value is monotonically non-increasing across
        all generations when elitism is active.

        Because elite individuals (those with the best fitness) are copied into each
        successive generation unchanged, and SphereEvaluator is deterministic, their
        fitness after re-evaluation is identical. Therefore the minimum fitness in the
        population can never be worse than the previous generation's minimum.

        :param elite_size: Number of elite individuals to preserve each generation.
        :returns: None
        :raises: None
        """
        inspector = FitnessCapturingInspector()
        island = minimal_island_factory(population_size=30, max_generations=5)
        island.set_elite_count(elite_size)
        island.set_seed(0)
        island.set_inspector(inspector)
        island.run()

        for i in range(len(inspector.best_values) - 1):
            assert inspector.best_values[i + 1] <= inspector.best_values[i], (
                f"Best fitness increased at generation {i + 1}: "
                f"{inspector.best_values[i]:.6f} -> {inspector.best_values[i + 1]:.6f}"
            )
