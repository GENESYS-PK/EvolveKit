"""
Unit tests for GaStatisticEngine – statistics calculation.

Covers: mean, median, standard-deviation, best/worst individual selection
for both MINIMUM and MAXIMUM extremum directions, and verifies that
best/worst individuals are deep-copied (not aliased).
"""

import numpy as np
import pytest

from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from tests.utils.factories.state_factories import statistic_engine_factory


VALUES_BASIC = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
VALUES_UNIFORM = [5.0, 5.0, 5.0, 5.0]
VALUES_SINGLE = [7.0]


class TestStatisticsDescriptives:
    """Tests for mean, median and standard-deviation after refresh()."""

    @pytest.mark.parametrize("values", [VALUES_BASIC, VALUES_UNIFORM, VALUES_SINGLE])
    def test_mean_matches_numpy_mean(self, values):
        """Mean reported by the engine must equal numpy's mean.

        :param values: Population fitness values.
        """
        engine, state = statistic_engine_factory(values)
        engine.refresh(state)

        assert engine.mean == pytest.approx(float(np.mean(values)))

    @pytest.mark.parametrize("values", [VALUES_BASIC, VALUES_UNIFORM, VALUES_SINGLE])
    def test_median_matches_numpy_median(self, values):
        """Median reported by the engine must equal numpy's median.

        :param values: Population fitness values.
        """
        engine, state = statistic_engine_factory(values)
        engine.refresh(state)

        assert engine.median == pytest.approx(float(np.median(values)))

    @pytest.mark.parametrize("values,expected_stdev", [
        (VALUES_BASIC, float(np.std(VALUES_BASIC))),
        (VALUES_UNIFORM, 0.0),
        (VALUES_SINGLE, 0.0),
    ])
    def test_stdev_matches_numpy_std(self, values, expected_stdev):
        """Standard deviation must equal numpy's population std (ddof=0).

        :param values: Population fitness values.
        :param expected_stdev: Expected standard deviation.
        """
        engine, state = statistic_engine_factory(values)
        engine.refresh(state)

        assert engine.stdev == pytest.approx(expected_stdev)


class TestStatisticsBestWorst:
    """Tests for best/worst individual selection for both optimisation directions."""

    @pytest.mark.parametrize("extremum,expected_best,expected_worst", [
        (GaExtremum.MINIMUM, min(VALUES_BASIC), max(VALUES_BASIC)),
        (GaExtremum.MAXIMUM, max(VALUES_BASIC), min(VALUES_BASIC)),
    ])
    def test_best_and_worst_values(self, extremum, expected_best, expected_worst):
        """best_indiv and worst_indiv must carry the correct fitness value for
        the given optimisation direction.

        :param extremum: Optimisation direction.
        :param expected_best: Expected best fitness value.
        :param expected_worst: Expected worst fitness value.
        """
        engine, state = statistic_engine_factory(VALUES_BASIC, extremum=extremum)
        engine.refresh(state)

        assert engine.best_indiv.value == pytest.approx(expected_best)
        assert engine.worst_indiv.value == pytest.approx(expected_worst)


class TestStatisticsDeepCopy:
    """Tests verifying that best/worst individuals are deep-copied, not aliased."""

    @pytest.mark.parametrize("attr,mutated_value", [
        ("best_indiv", 999.0),
        ("worst_indiv", 0.0),
    ])
    def test_stored_individual_is_independent_of_population(self, attr, mutated_value):
        """Mutating every individual in the population must not change the value
        already stored in best_indiv / worst_indiv.

        :param attr: Attribute name to check ('best_indiv' or 'worst_indiv').
        :param mutated_value: Value assigned to the population to detect aliasing.
        """
        values = [1.0, 5.0, 3.0]
        engine, state = statistic_engine_factory(values, extremum=GaExtremum.MINIMUM)
        engine.refresh(state)

        value_before = getattr(engine, attr).value

        for ind in state.current_population:
            ind.value = mutated_value

        assert getattr(engine, attr).value == pytest.approx(value_before)
