"""
Factory helpers for creating configured GaState instances in tests.
"""

from typing import List

from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaStatisticEngine import GaStatisticEngine
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from tests.utils.mocks.mock_objects import MockEvaluator
from tests.utils.factories.individual_factories import create_individual


def population_with_values_factory(values: list[float]) -> List[GaIndividual]:
    """
    Create a list of GaIndividuals whose real_chrom and value field both reflect
    the provided scalar *values*.

    This is the building block used by :func:`state_with_population_factory` and
    can also be used directly inside tests that need to replace or update a
    state's population mid-test.

    :param values: Scalar fitness values; each produces one individual.
    :returns: List of GaIndividual instances with pre-set fitness values.
    """
    individuals = []
    for v in values:
        ind = create_individual([v])
        ind.value = v
        individuals.append(ind)
    return individuals


def state_with_population_factory(
    values: list[float],
    extremum: GaExtremum = GaExtremum.MINIMUM,
) -> GaState:
    """
    Build a GaState whose current_population contains individuals with fitness
    values matching *values*. Attaches a MockEvaluator with the given extremum.

    :param values: Fitness values for each individual in the population.
    :param extremum: Optimisation direction (MINIMUM or MAXIMUM).
    :returns: Configured GaState ready for statistic-engine operations.
    """
    state = GaState()
    state.evaluator = MockEvaluator(dim=len(values), extremum_type=extremum)
    state.current_population = population_with_values_factory(values)
    return state


def configured_state_factory() -> GaState:
    """
    Return a GaState configured with non-default scalar parameters, a populated
    current_population, and statistics that have been advanced once.

    Useful for serialisation / deepcopy tests that need meaningful data.

    :returns: A fully configured GaState instance.
    """
    state = GaState()
    state.seed = 99
    state.crossover_prob = 0.8
    state.mutation_prob = 0.05
    state.max_generations = 300
    state.population_size = 50
    state.elite_size = 5
    state.real_clamp_strategy = GaClampStrategy.CLAMP
    state.evaluator = MockEvaluator(dim=3)
    state.current_population = population_with_values_factory([float(i) for i in range(5)])
    state.statistic_engine.start(state)
    state.statistic_engine.advance(state)
    return state


def statistic_engine_factory(
    values: list[float],
    extremum: GaExtremum = GaExtremum.MINIMUM,
    started: bool = True,
) -> tuple["GaStatisticEngine", "GaState"]:
    """
    Build a GaStatisticEngine that has been started against a state whose
    population carries the given fitness *values*.

    Returns both the engine and the backing state so callers can call
    ``advance`` / ``refresh`` inside their tests.

    :param values: Fitness values for each individual in the population.
    :param extremum: Optimisation direction (MINIMUM or MAXIMUM).
    :param started: Whether to call ``engine.start(state)`` before returning.
    :returns: Tuple ``(GaStatisticEngine, GaState)``.
    """
    state = state_with_population_factory(values, extremum)
    engine = state.statistic_engine
    if started:
        engine.start(state)
    return engine, state
