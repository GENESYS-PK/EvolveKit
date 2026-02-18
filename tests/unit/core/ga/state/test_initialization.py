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

    def test_current_population_is_empty_list(self):
        """Test that current_population defaults to an empty list.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.current_population == []

    def test_selected_population_is_empty_list(self):
        """Test that selected_population defaults to an empty list.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.selected_population == []

    def test_offspring_population_is_empty_list(self):
        """Test that offspring_population defaults to an empty list.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.offspring_population == []

    def test_elite_population_is_empty_list(self):
        """Test that elite_population defaults to an empty list.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.elite_population == []

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

    def test_crossover_prob_is_zero(self):
        """Test that crossover_prob defaults to 0.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.crossover_prob == 0

    def test_mutation_prob_is_zero(self):
        """Test that mutation_prob defaults to 0.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.mutation_prob == 0

    def test_max_generations_is_zero(self):
        """Test that max_generations defaults to 0.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.max_generations == 0

    def test_population_size_is_zero(self):
        """Test that population_size defaults to 0.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.population_size == 0

    def test_elite_size_is_zero(self):
        """Test that elite_size defaults to 0.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert state.elite_size == 0

    def test_statistic_engine_is_created(self):
        """Test that statistic_engine is initialized as a GaStatisticEngine instance.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert isinstance(state.statistic_engine, GaStatisticEngine)


class TestGaStateSeedInitialization:
    """Test auto-generated seed behavior on state creation."""

    def test_seed_is_positive_integer(self):
        """Test that the auto-generated seed is a positive integer.

        :returns: None
        :raises: None
        """
        state = GaState()

        assert isinstance(state.seed, int)
        assert state.seed >= 1

    def test_seed_is_within_valid_range(self):
        """Test that the auto-generated seed is within [1, 2^32 - 1].

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
    """Test that parameters can be assigned and stored correctly."""

    def test_assign_crossover_prob(self):
        """Test assigning a valid crossover probability.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.crossover_prob = 0.85

        assert state.crossover_prob == 0.85

    def test_assign_mutation_prob(self):
        """Test assigning a valid mutation probability.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.mutation_prob = 0.05

        assert state.mutation_prob == 0.05

    def test_assign_max_generations(self):
        """Test assigning max generations count.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.max_generations = 500

        assert state.max_generations == 500

    def test_assign_population_size(self):
        """Test assigning population size.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.population_size = 200

        assert state.population_size == 200

    def test_assign_elite_size(self):
        """Test assigning elite size.

        :returns: None
        :raises: None
        """
        state = GaState()
        state.elite_size = 10

        assert state.elite_size == 10

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
