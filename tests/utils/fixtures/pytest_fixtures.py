import pytest

from ..factories.individual_factories import random_individual_factory
from ..factories.population_factories import population_factory
from ..factories.evaluator_factories import evaluator_factory
from ..factories.state_factories import (
    state_with_population_factory,
    configured_state_factory,
)


@pytest.fixture
def individual_factory():
    """
    Fixture that returns the random_individual_factory function.
    Usage: individual = individual_factory(dimension=5, seed=42)
    """
    return random_individual_factory


@pytest.fixture
def population_factory_fixture():
    """
    Fixture that returns the population_factory function.
    Usage: population = population_factory_fixture(dimension=10, population_size=50, seed=42)
    """
    return population_factory


@pytest.fixture
def evaluator_factory_fixture():
    """
    Fixture that returns the evaluator_factory function.
    Usage: evaluator = evaluator_factory_fixture("sphere", dimension=3)
    """
    return evaluator_factory


@pytest.fixture
def state_with_population_fixture():
    """
    Fixture that returns the state_with_population_factory function.
    Usage: state = state_with_population_fixture([1.0, 2.0, 3.0], GaExtremum.MINIMUM)
    """
    return state_with_population_factory


@pytest.fixture
def configured_state_fixture():
    """
    Fixture that returns a fully configured GaState instance (non-default parameters,
    populated current_population, statistics advanced once).
    Usage: state = configured_state_fixture()
    """
    return configured_state_factory
