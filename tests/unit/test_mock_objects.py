import numpy as np
import pytest
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs

from tests.utils.mocks.mock_objects import (
    MockEvaluator,
    MockOperator,
    TestDataGenerator,
    TestScenarios,
)


class TestMockEvaluator:
    """Test the MockEvaluator class."""

    def test_mock_evaluator_creation(self):
        """
        Test creating a mock evaluator with specific parameters.

        Creates MockEvaluator with dim=5 and constant_value=42.0
        and verifies the initialization values are correctly set.

        :returns: None
        :raises: None
        """
        evaluator = MockEvaluator(dim=5, constant_value=42.0)
        assert evaluator.dim == 5
        assert evaluator.constant_value == 42.0
        assert evaluator.evaluation_count == 0

    def test_mock_evaluation(self):
        """
        Test mock evaluation functionality.

        Evaluates an individual with chromosome [1.0, 2.0, 3.0]
        using MockEvaluator and verifies:
        - Result equals constant_value (10.0)
        - Evaluation count increments to 1
        - Chromosome is stored in evaluation history

        :returns: None
        :raises: None
        """
        evaluator = MockEvaluator(constant_value=10.0)
        individual = GaIndividual(real_chrom=np.array([1.0, 2.0, 3.0]))
        args = GaEvaluatorArgs(individual)

        result = evaluator.evaluate(args)
        assert result == 10.0
        assert evaluator.evaluation_count == 1
        assert len(evaluator.evaluation_history) == 1
        np.testing.assert_array_equal(evaluator.evaluation_history[0], [1.0, 2.0, 3.0])

    def test_mock_domain(self):
        """
        Test mock domain functionality.

        Creates MockEvaluator with dim=3 and verifies
        the real_domain returns 3 bounds all set to (0.0, 1.0).

        :returns: None
        :raises: None
        """
        evaluator = MockEvaluator(dim=3)
        domain = evaluator.real_domain()
        assert len(domain) == 3
        assert all(bounds == (0.0, 1.0) for bounds in domain)

    def test_mock_reset(self):
        """
        Test resetting mock evaluator state.

        Performs an evaluation to increment counters, then resets
        and verifies evaluation_count and history are cleared.

        :returns: None
        :raises: None
        """
        evaluator = MockEvaluator()
        individual = GaIndividual(real_chrom=np.array([1.0]))
        args = GaEvaluatorArgs(individual)

        evaluator.evaluate(args)
        assert evaluator.evaluation_count == 1

        evaluator.reset()
        assert evaluator.evaluation_count == 0
        assert len(evaluator.evaluation_history) == 0


class TestMockOperator:
    """Test the MockOperator class."""

    def test_mock_operator_creation(self):
        """
        Test creating a mock operator with specific name.

        Creates MockOperator with operation_name='test_op'
        and verifies initialization values are correctly set.

        :returns: None
        :raises: None
        """
        operator = MockOperator("test_op")
        assert operator.operation_name == "test_op"
        assert operator.application_count == 0

    def test_mock_apply(self):
        """
        Test mock operator application (identity operation).

        Applies MockOperator to an individual and verifies:
        - Result is the same individual (identity)
        - Application count increments to 1
        - Last input is stored correctly

        :returns: None
        :raises: None
        """
        operator = MockOperator()
        individual = GaIndividual(real_chrom=np.array([1.0, 2.0]))

        result = operator.apply(individual)
        assert result is individual
        assert operator.application_count == 1
        assert operator.last_input is individual

    def test_mock_reset(self):
        """
        Test resetting mock operator state.

        Performs an apply operation to increment counters, then resets
        and verifies application_count and last_input are cleared.

        :returns: None
        :raises: None
        """
        operator = MockOperator()
        operator.apply("test")
        assert operator.application_count == 1

        operator.reset()
        assert operator.application_count == 0
        assert operator.last_input is None


class TestTestDataGenerator:
    """Test the TestDataGenerator class."""

    def test_linear_population(self):
        """
        Test creating linearly spaced population.

        Creates 5 individuals with 3D chromosomes starting at 1.0
        with step 0.5, and verifies correct population size,
        chromosome dimensions, and linear spacing.

        :returns: None
        :raises: None
        """
        population = TestDataGenerator.create_linear_population(
            5, 3, start=1.0, step=0.5
        )
        assert len(population) == 5
        assert all(len(ind.real_chrom) == 3 for ind in population)

        for i, individual in enumerate(population):
            expected_value = 1.0 + i * 0.5
            assert np.allclose(individual.real_chrom, [expected_value] * 3)

    def test_diverse_population(self):
        """
        Test creating diverse population within bounds.

        Creates 10 individuals with 3D chromosomes within bounds
        [(0,1), (-1,1), (2,3)] and verifies all individuals
        respect their respective dimensional bounds.

        :returns: None
        :raises: None
        """
        bounds = [(0.0, 1.0), (-1.0, 1.0), (2.0, 3.0)]
        population = TestDataGenerator.create_diverse_population(10, 3, bounds)
        assert len(population) == 10

        for individual in population:
            assert len(individual.real_chrom) == 3
            assert 0.0 <= individual.real_chrom[0] <= 1.0
            assert -1.0 <= individual.real_chrom[1] <= 1.0
            assert 2.0 <= individual.real_chrom[2] <= 3.0

    def test_clustered_population(self):
        """
        Test creating clustered population.

        Creates 15 individuals with 2D chromosomes in 3 clusters
        and verifies population size, chromosome dimensions,
        and bounds compliance [0,1] for all values.

        :returns: None
        :raises: None
        """
        population = TestDataGenerator.create_clustered_population(
            15, 2, num_clusters=3
        )
        assert len(population) == 15
        assert all(len(ind.real_chrom) == 2 for ind in population)
        assert all(
            np.all((ind.real_chrom >= 0.0) & (ind.real_chrom <= 1.0))
            for ind in population
        )

    def test_known_optima_data(self):
        """
        Test creating known optima data for benchmark functions.

        Creates optima data for 3D problems and verifies:
        - Sphere optimum is at [0.0, 0.0, 0.0]
        - Rosenbrock optimum is at [1.0, 1.0, 1.0]

        :returns: None
        :raises: None
        """
        optima = TestDataGenerator.create_known_optima_data(3)

        assert "sphere_optimum" in optima
        assert "rosenbrock_optimum" in optima

        np.testing.assert_array_equal(
            optima["sphere_optimum"].real_chrom, [0.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            optima["rosenbrock_optimum"].real_chrom, [1.0, 1.0, 1.0]
        )


class TestTestScenarios:
    """Test the TestScenarios class."""

    def test_optimization_convergence_scenario(self):
        """
        Test optimization convergence scenario creation.

        Creates scenario with SphereEvaluator and 3 individuals,
        verifies scenario contains required keys and best fitness
        is 0.0 (from individual at origin).

        :returns: None
        :raises: None
        """
        from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator

        evaluator = SphereEvaluator(dim=2)
        population = [
            GaIndividual(real_chrom=np.array([0.0, 0.0])),
            GaIndividual(real_chrom=np.array([1.0, 1.0])),
            GaIndividual(real_chrom=np.array([2.0, 2.0])),
        ]

        scenario = TestScenarios.optimization_convergence_scenario(
            evaluator, population
        )

        assert "initial_population" in scenario
        assert "initial_fitness" in scenario
        assert "best_individual" in scenario
        assert "best_fitness" in scenario
        assert len(scenario["initial_fitness"]) == 3
        assert scenario["best_fitness"] == 0.0

    def test_operator_stress_test_scenario(self):
        """
        Test operator stress test scenario creation.

        Creates stress test scenario with MockOperator and 2 individuals,
        verifies scenario contains operator, population, and edge cases
        including identical and diverse parent combinations.

        :returns: None
        :raises: None
        """
        mock_operator = MockOperator()
        population = [
            GaIndividual(real_chrom=np.array([0.0, 0.0])),
            GaIndividual(real_chrom=np.array([1.0, 1.0])),
        ]

        scenario = TestScenarios.operator_stress_test_scenario(
            mock_operator, population
        )

        assert "operator" in scenario
        assert "test_population" in scenario
        assert "edge_cases" in scenario
        assert "identical_parents" in scenario["edge_cases"]
        assert "diverse_parents" in scenario["edge_cases"]
