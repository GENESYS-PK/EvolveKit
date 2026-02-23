"""
Unit tests for GaIsland initialization.

Tests the default parameter values set on construction, the presence of
required default operators, and that __verify raises appropriate exceptions
for invalid configurations before run() begins.
"""

import pytest
from unittest.mock import MagicMock

from evolvekit.core.Ga.GaIsland import GaIsland
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.operators.Ga.binary.crossover.OnePointCrossover import (
    OnePointCrossover,
)
from tests.utils.factories.island_factories import minimal_island_factory


class TestGaIslandDefaultValues:
    """Test that GaIsland overrides GaState defaults with sensible initial values."""

    @pytest.mark.parametrize("field, expected", [
        ("population_size", 100),
        ("elite_size", 0),
        ("max_generations", 200),
        ("seed", 0),
    ])
    def test_integer_parameter_defaults(self, field, expected):
        """Test that integer configuration fields are set to expected defaults.

        :param field: Attribute name to inspect.
        :param expected: Expected default value.
        :returns: None
        :raises: None
        """
        island = GaIsland()

        assert getattr(island, field) == expected

    @pytest.mark.parametrize("field, expected", [
        ("crossover_prob", 0.9),
        ("mutation_prob", 0.1),
    ])
    def test_probability_defaults(self, field, expected):
        """Test that probability fields are set to expected defaults.

        :param field: Attribute name to inspect.
        :param expected: Expected default value.
        :returns: None
        :raises: None
        """
        island = GaIsland()

        assert getattr(island, field) == pytest.approx(expected)

    def test_clamp_strategy_defaults_to_none(self):
        """Test that real_clamp_strategy is set to GaClampStrategy.NONE by default.

        :returns: None
        :raises: None
        """
        island = GaIsland()

        assert island.real_clamp_strategy == GaClampStrategy.NONE

    def test_inspector_defaults_to_none(self):
        """Test that no inspector is attached on construction.

        :returns: None
        :raises: None
        """
        island = GaIsland()

        assert island.inspector is None

    @pytest.mark.parametrize("operator_field", [
        "selection",
        "real_crossover",
        "real_mutation",
        "bin_crossover",
        "bin_mutation",
    ])
    def test_default_operators_are_set(self, operator_field):
        """Test that all genetic operators are wired up by default.

        :param operator_field: Name of the operator attribute.
        :returns: None
        :raises: None
        """
        island = GaIsland()

        assert getattr(island, operator_field) is not None

    def test_is_subclass_of_ga_state(self):
        """Test that GaIsland inherits from GaState.

        :returns: None
        :raises: None
        """
        assert isinstance(GaIsland(), GaState)


class TestGaIslandParameterValidation:
    """Test that __verify raises appropriate exceptions for invalid configurations."""

    def test_run_raises_when_evaluator_is_missing(self):
        """Test that a TypeError is raised when no evaluator has been set.

        :returns: None
        :raises: None
        """
        island = GaIsland()

        with pytest.raises(TypeError):
            island.run()

    def test_run_raises_when_selection_is_none(self):
        """Test that a TypeError is raised when selection operator is removed.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        island.selection = None

        with pytest.raises(TypeError):
            island.run()

    @pytest.mark.parametrize("field, invalid_value", [
        ("population_size", 0),
        ("population_size", -5),
        ("elite_size", -1),
        ("max_generations", 0),
        ("crossover_prob", -0.1),
        ("crossover_prob", 1.1),
        ("mutation_prob", -0.1),
        ("mutation_prob", 1.1),
    ])
    def test_run_raises_on_invalid_parameter(self, field, invalid_value):
        """Test that a ValueError is raised for each out-of-range configuration.

        :param field: Attribute name to override.
        :param invalid_value: Invalid value to assign.
        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        setattr(island, field, invalid_value)

        with pytest.raises(ValueError):
            island.run()

    def test_run_raises_when_real_crossover_has_wrong_category(self):
        """Test that a TypeError is raised when an operator of the wrong category
        is assigned to the real_crossover slot.

        :returns: None
        :raises: None
        """
        island = minimal_island_factory()
        island.real_crossover = OnePointCrossover()  # BIN_CROSSOVER, not REAL_CROSSOVER

        with pytest.raises(TypeError):
            island.run()


class TestGaIslandSetOperator:
    """Test that set_operator dispatches to the correct attribute slot."""

    @pytest.mark.parametrize("op_field, category", [
        ("selection", GaOpCategory.SELECTION),
        ("real_crossover", GaOpCategory.REAL_CROSSOVER),
        ("real_mutation", GaOpCategory.REAL_MUTATION),
        ("bin_crossover", GaOpCategory.BIN_CROSSOVER),
        ("bin_mutation", GaOpCategory.BIN_MUTATION),
    ])
    def test_set_operator_routes_to_correct_slot(self, op_field, category):
        """Test that set_operator assigns the operator to the slot matching its category.

        :param op_field: Expected target attribute name.
        :param category: Operator category the mock will report.
        :returns: None
        :raises: None
        """
        island = GaIsland()
        mock_op = MagicMock()
        mock_op.category.return_value = category

        island.set_operator(mock_op)

        assert getattr(island, op_field) is mock_op
