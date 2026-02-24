"""
Factory helpers for creating configured GaIsland instances in tests.
"""

from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator


def minimal_island_factory(
    dim: int = 15,
    max_generations: int = 2,
    population_size: int = 10,
) -> GaIsland:
    """
    Create a minimal, ready-to-run GaIsland configured with a SphereEvaluator.

    The returned island keeps all default operators and only overrides the
    parameters needed for fast test execution (small population, few generations).

    :param dim: Number of real-valued dimensions for the SphereEvaluator.
    :param max_generations: Upper bound on the number of evolution cycles.
    :param population_size: Number of individuals in the population.
    :returns: A configured GaIsland that can be started with :meth:`run`.
    """
    island = GaIsland()
    island.set_evaluator(SphereEvaluator(dim=dim))
    island.set_max_generations(max_generations)
    island.set_population_size(population_size)
    return island
