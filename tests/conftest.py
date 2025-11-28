import pytest
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def configure_numpy():
    np.random.seed(42)
