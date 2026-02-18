import pytest
import numpy as np

# Import fixtures to make them available across all test modules
from tests.utils import (
    individual_factory,
    population_factory_fixture, 
    evaluator_factory_fixture,
    state_with_population_fixture,
    configured_state_fixture,
)


@pytest.fixture(scope="session", autouse=True)
def configure_numpy():
    np.random.seed(42)
