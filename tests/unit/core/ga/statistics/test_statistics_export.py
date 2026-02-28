"""
Unit tests for GaStatistics – data format and export.

Covers: default field values, correct types after population evaluation,
verification of all declared dataclass fields, and dictionary serialisation
via ``dataclasses.asdict``.
"""

import dataclasses

import numpy as np
import pytest

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaStatistics import GaStatistics
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from tests.utils.factories.state_factories import statistic_engine_factory


EXPECTED_FIELDS = {
    "generation",
    "stagnation",
    "mean",
    "median",
    "stdev",
    "best_indiv",
    "worst_indiv",
    "start_time",
    "last_time",
}


class TestGaStatisticsDefaultValues:
    """Every field of a freshly constructed GaStatistics must equal its documented
    default.
    """

    @pytest.mark.parametrize("field,expected", [
        ("generation", 0),
        ("stagnation", 0),
        ("mean", 0.0),
        ("median", 0.0),
        ("stdev", 0.0),
        ("start_time", 0.0),
        ("last_time", 0.0),
        ("best_indiv", None),
        ("worst_indiv", None),
    ])
    def test_default_field_value(self, field, expected):
        """Field ``field`` must equal ``expected`` on a fresh GaStatistics instance.

        :param field: Attribute name.
        :param expected: Expected default value.
        """
        stats = GaStatistics()
        assert getattr(stats, field) == expected


class TestGaStatisticsFieldSchema:
    """Tests that GaStatistics is a proper dataclass with all expected fields."""

    def test_ga_statistics_is_a_dataclass(self):
        """GaStatistics must be decorated with @dataclass."""
        assert dataclasses.is_dataclass(GaStatistics)

    def test_ga_statistics_has_all_expected_dataclass_fields(self):
        """dataclasses.fields() must include every documented field name."""
        field_names = {f.name for f in dataclasses.fields(GaStatistics)}
        missing = EXPECTED_FIELDS - field_names
        assert not missing, f"Missing dataclass fields: {missing}"

    @pytest.mark.parametrize("field", sorted(EXPECTED_FIELDS))
    def test_field_is_accessible_on_fresh_instance(self, field):
        """Every documented field must be accessible as an attribute.

        :param field: Attribute name to check.
        """
        assert hasattr(GaStatistics(), field)


class TestGaStatisticsFieldTypes:
    """Every statistics field must have the correct Python type after advance()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        engine, state = statistic_engine_factory(
            [1.0, 3.0, 5.0], extremum=GaExtremum.MINIMUM
        )
        engine.advance(state)
        self.engine = engine

    @pytest.mark.parametrize("field", ["generation", "stagnation"])
    def test_counter_fields_are_int(self, field):
        """generation and stagnation must be int instances after advance().

        :param field: Counter field name.
        """
        assert isinstance(getattr(self.engine, field), int)

    @pytest.mark.parametrize("field", ["mean", "median", "stdev", "start_time", "last_time"])
    def test_numeric_fields_are_float(self, field):
        """Numeric statistic fields must be float instances after advance().

        :param field: Numeric field name.
        """
        assert isinstance(getattr(self.engine, field), float)

    @pytest.mark.parametrize("field", ["best_indiv", "worst_indiv"])
    def test_individual_fields_are_ga_individual(self, field):
        """best_indiv and worst_indiv must be GaIndividual instances after advance().

        :param field: Individual field name.
        """
        assert isinstance(getattr(self.engine, field), GaIndividual)


class TestGaStatisticsAsdict:
    """Tests that GaStatistics / GaStatisticEngine correctly serialise via asdict."""

    def test_asdict_contains_all_expected_keys(self):
        """dataclasses.asdict() on a default GaStatistics must contain every
        documented field as a key.
        """
        exported = dataclasses.asdict(GaStatistics())
        missing = EXPECTED_FIELDS - exported.keys()
        assert not missing, f"Keys missing from asdict() output: {missing}"

    def test_asdict_values_match_manually_set_attributes(self):
        """Values in the dict returned by asdict() must reflect manually assigned
        attribute values.
        """
        stats = GaStatistics()
        stats.generation = 7
        stats.stagnation = 3
        stats.mean = 2.5

        exported = dataclasses.asdict(stats)

        assert exported["generation"] == 7
        assert exported["stagnation"] == 3
        assert exported["mean"] == pytest.approx(2.5)

    def test_asdict_after_engine_advance_has_correct_numeric_fields(self):
        """A dict exported after engine.advance() must contain numerically correct
        generation, mean, and stdev for the evaluated population.
        """
        values = [2.0, 4.0, 6.0]
        engine, state = statistic_engine_factory(values)
        engine.advance(state)

        exported = dataclasses.asdict(engine)

        assert exported["generation"] == 1
        assert exported["mean"] == pytest.approx(float(np.mean(values)))
        assert exported["stdev"] == pytest.approx(float(np.std(values)))
