"""
Unit tests for GaState initialization.

Tests proper default values set during construction, type checking
of initialized fields, and independence of multiple GaState instances.
"""

import pytest

from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.GaStatisticEngine import GaStatisticEngine
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from tests.utils.factories.individual_factories import create_individual
from tests.utils.mocks.mock_objects import MockEvaluator


class TestGaStateDefaultValues:
    """Test that all fields are correctly initialized to their default values."""

    @pytest.mark.parametrize("population_field", [
        "current_population",
        "selected_population",
        "offspring_population",
        "elite_population",
    ])
    def test_population_fields_default_to_empty_list(self, population_field):
        """Test that all population fields default to an empty list.

        :param population_field: Name of the population attribute to check
        :returns: None
        :raises: None
        """
        state = GaState()

        assert getattr(state, population_field) == []

    def test_evaluator_is_none(self):
        """Test that evaluator defaults to None.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.evaluator is None

    def test_real_clamp_strategy_is_none(self):
        """Test that real_clamp_strategy defaults to GaClampStrategy.NONE.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.real_clamp_strategy == GaClampStrategy.NONE

    @pytest.mark.parametrize("numeric_field", [
        "crossover_prob",
        "mutation_prob",
        "max_generations",
        "population_size",
        "elite_size",
    ])
    def test_numeric_fields_default_to_zero(self, numeric_field):
        """Test that all numeric configuration fields default to 0.

        :param numeric_field: Name of the numeric attribute to check
        :returns: None
        :raises: None
        """
        state = GaState()

        assert getattr(state, numeric_field) == 0

    def test_statistic_engine_is_created(self):
        """Test that statistic_engine is initialized as a GaStatisticEngine instance.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert isinstance(state.statistic_engine, GaStatisticEngine)


class TestGaStateSeedInitialization:
    """Test auto-generated seed behavior on state creation."""

    def test_seed_is_within_valid_range(self):
        """Test that the auto-generated seed is a positive integer within [1, 2^32 - 1].

        :returns: None
        :raises: None
        """
        state = GaState()

        assert 1 <= state.seed <= (2**32 - 1)

    def test_multiple_instances_have_independent_seeds(self):
        """Test that two GaState instances are likely to have different seeds.

        Running this over several instances gives high confidence they are independent.

        :returns: None
        :raises: None
        """
        seeds = {GaState().seed for _ in range(20)}

        # With 20 samples from a 2^32 range, collision probability is negligible
        assert len(seeds) > 1

    def test_seed_can_be_overwritten(self):
        """Test that seed can be manually set after initialization.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.seed = 12345

        assert state.seed == 12345


class TestGaStateInstanceIndependence:
    """Test that multiple GaState instances do not share mutable state."""

    def test_population_lists_are_independent(self):
        """Test that modifying one state's population does not affect another.

        :returns: None
        :raises: None
        """
        state_a = GaState()
        state_b = GaState()
        individual = create_individual([1.0, 2.0])

        state_a.current_population.append(individual)

        assert len(state_a.current_population) == 1
        assert len(state_b.current_population) == 0

    def test_statistic_engines_are_independent(self):
        """Test that each GaState instance has its own statistic engine.

        :returns: None
        :raises: None
        """
        state_a = GaState()
        state_b = GaState()

        assert state_a.statistic_engine is not state_b.statistic_engine

    def test_parameters_are_independent(self):
        """Test that parameter changes in one instance do not affect another.

        :returns: None
        :raises: None
        """
        state_a = GaState()
        state_b = GaState()

        state_a.crossover_prob = 0.9
        state_a.mutation_prob = 0.1
        state_a.max_generations = 100

        assert state_b.crossover_prob == 0
        assert state_b.mutation_prob == 0
        assert state_b.max_generations == 0


class TestGaStateParameterAssignment:
    """Test that EvolveKit-specific types can be assigned and stored correctly."""

    def test_assign_evaluator(self):
        """Test assigning an evaluator to the state.

        :returns: None
        :raises: None
        """
        state = GaState()
        evaluator = MockEvaluator(dim=5)
        state.evaluator = evaluator

        assert state.evaluator is evaluator

    @pytest.mark.parametrize("strategy", list(GaClampStrategy))
    def test_assign_all_clamp_strategies(self, strategy):
        """Test assigning each available GaClampStrategy value.

        :param strategy: A GaClampStrategy enum value
        :returns: None
        :raises: None
        """
        state = GaState()
        state.real_clamp_strategy = strategy

        assert state.real_clamp_strategy == strategy
