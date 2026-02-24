import numpy as np
from typing import List, Tuple, Any

from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.enums.GaAction import GaAction
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


class MockEvaluator(GaEvaluator):
    """Mock evaluator for testing purposes."""

    def __init__(self, dim: int = 5, constant_value: float = 1.0, extremum_type: GaExtremum = GaExtremum.MINIMUM):
        """
        Initialize mock evaluator.

        :param dim: Problem dimension
        :param constant_value: Constant value to return for all evaluations
        :param extremum_type: Optimization direction (MINIMUM or MAXIMUM)
        """
        self.dim = dim
        self.constant_value = constant_value
        self.extremum_type = extremum_type
        self.evaluation_count = 0
        self.evaluation_history = []

    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Mock evaluation function.

        :param args: Evaluation arguments
        :return: Constant value
        """
        self.evaluation_count += 1
        chromosome = args.real_chrom.copy()
        self.evaluation_history.append(chromosome)
        return self.constant_value

    def extremum(self) -> GaExtremum:
        """
        Returns the optimization direction.

        :return: GaExtremum enum value
        """
        return self.extremum_type

    def real_domain(self) -> List[Tuple[float, float]]:
        """Return mock domain bounds."""
        return [(0.0, 1.0) for _ in range(self.dim)]

    def reset(self):
        """Reset evaluation counter and history."""
        self.evaluation_count = 0
        self.evaluation_history = []


class MockBinaryEvaluator(GaEvaluator):
    """Mock evaluator that uses binary chromosome - counts 1s in bit string."""
    
    def __init__(self, bin_len: int = 10):
        """
        Initialize binary evaluator.
        
        :param bin_len: Length of binary chromosome
        """
        self._bin_len = bin_len
    
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Count 1s in binary chromosome.
        
        :param args: Evaluation arguments
        :return: Sum of 1s in binary chromosome
        """
        bits = np.unpackbits(args.bin_chrom)
        return float(np.sum(bits[:self._bin_len]))
    
    def extremum(self) -> GaExtremum:
        """
        Returns MAXIMUM (we want to maximize 1s).
        
        :return: GaExtremum.MAXIMUM
        """
        return GaExtremum.MAXIMUM
    
    def bin_length(self) -> int:
        """
        Returns binary chromosome length.
        
        :return: Binary length
        """
        return self._bin_len


class TerminatingInspector(GaInspector):
    """Inspector that immediately signals termination on the first inspect call."""

    def inspect(self, stats) -> GaAction:
        """Return TERMINATE unconditionally.

        :param stats: Current simulation statistics (unused).
        :returns: GaAction.TERMINATE
        """
        return GaAction.TERMINATE


class AfterNGenerationsInspector(GaInspector):
    """Inspector that terminates the simulation after exactly N generations."""

    def __init__(self, n: int):
        """
        :param n: Number of generations to allow before terminating.
        """
        self._n = n

    def inspect(self, stats) -> GaAction:
        """Return TERMINATE once generation count reaches N.

        :param stats: Current simulation statistics.
        :returns: GaAction.TERMINATE when generation >= N, else GaAction.CONTINUE.
        """
        if stats.generation >= self._n:
            return GaAction.TERMINATE
        return GaAction.CONTINUE


class FitnessCapturingInspector(GaInspector):
    """Inspector that records best_indiv.value at every generation without interfering."""

    def __init__(self):
        self.best_values: list[float] = []

    def inspect(self, stats) -> GaAction:
        """Append current best fitness and let the run continue.

        :param stats: Current simulation statistics.
        :returns: GaAction.CONTINUE always.
        """
        self.best_values.append(stats.best_indiv.value)
        return GaAction.CONTINUE


class MockMixedEvaluator(GaEvaluator):
    """Mock evaluator that uses both real and binary chromosomes."""
    
    def __init__(self, real_dim: int = 3, bin_len: int = 5):
        """
        Initialize mixed chromosome evaluator.
        
        :param real_dim: Number of real-valued dimensions
        :param bin_len: Length of binary chromosome
        """
        self._real_dim = real_dim
        self._bin_len = bin_len
    
    def evaluate(self, args: GaEvaluatorArgs) -> float:
        """
        Evaluate using both real and binary parts (sphere + binary sum).
        
        :param args: Evaluation arguments
        :return: Sum of squared real values plus binary 1s
        """
        real_part = np.sum(np.square(args.real_chrom[:self._real_dim]))
        bits = np.unpackbits(args.bin_chrom)
        bin_part = np.sum(bits[:self._bin_len])
        return float(real_part + bin_part)
    
    def extremum(self) -> GaExtremum:
        """
        Returns MINIMUM.
        
        :return: GaExtremum.MINIMUM
        """
        return GaExtremum.MINIMUM
    
    def real_domain(self) -> List[Tuple[float, float]]:
        """
        Returns domain bounds for real-valued dimensions.
        
        :return: List of (min, max) tuples
        """
        return [(-5.0, 5.0)] * self._real_dim
    
    def bin_length(self) -> int:
        """
        Returns binary chromosome length.
        
        :return: Binary length
        """
        return self._bin_len


class MockOperator:
    """Mock operator for testing purposes."""

    def __init__(self, operation_name: str = "mock_operation"):
        """
        Initialize mock operator.

        :param operation_name: Name of the operation
        """
        self.operation_name = operation_name
        self.application_count = 0
        self.last_input = None

    def apply(self, *args, **kwargs) -> Any:
        """
        Mock apply method.

        :param args: Arguments
        :param kwargs: Keyword arguments
        :return: First argument (identity operation)
        """
        self.application_count += 1
        if args:
            self.last_input = args[0]
            return args[0]
        return None

    def reset(self):
        """Reset application counter."""
        self.application_count = 0
        self.last_input = None


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def create_linear_population(
        size: int, dim: int, start: float = 0.0, step: float = 0.1
    ) -> List[GaIndividual]:
        """
        Create a population with linearly spaced values.

        :param size: Population size
        :param dim: Chromosome dimension
        :param start: Starting value
        :param step: Step size between individuals
        :return: List of individuals
        """
        population = []
        for i in range(size):
            values = [start + i * step] * dim
            individual = GaIndividual(real_chrom=np.array(values))
            population.append(individual)
        return population

    @staticmethod
    def create_diverse_population(
        size: int, dim: int, bounds: List[Tuple[float, float]]
    ) -> List[GaIndividual]:
        """
        Create a diverse population within given bounds.

        :param size: Population size
        :param dim: Chromosome dimension
        :param bounds: List of (min, max) bounds for each dimension
        :return: List of individuals
        """
        population = []
        for _ in range(size):
            chromosome = np.zeros(dim)
            for i, (min_val, max_val) in enumerate(bounds):
                chromosome[i] = np.random.uniform(min_val, max_val)
            individual = GaIndividual(real_chrom=chromosome)
            population.append(individual)
        return population

    @staticmethod
    def create_clustered_population(
        size: int, dim: int, num_clusters: int = 3
    ) -> List[GaIndividual]:
        """
        Create a population with clustered individuals.

        :param size: Population size
        :param dim: Chromosome dimension
        :param num_clusters: Number of clusters
        :return: List of individuals
        """
        population = []
        individuals_per_cluster = size // num_clusters

        for cluster in range(num_clusters):
            # Create cluster center
            center = np.random.rand(dim)

            for _ in range(individuals_per_cluster):
                # Add noise around cluster center
                noise = np.random.normal(0, 0.1, dim)
                chromosome = center + noise
                # Clamp to [0, 1]
                chromosome = np.clip(chromosome, 0.0, 1.0)
                individual = GaIndividual(real_chrom=chromosome)
                population.append(individual)

        # Add remaining individuals if size is not divisible by num_clusters
        remaining = size - len(population)
        for _ in range(remaining):
            chromosome = np.random.rand(dim)
            individual = GaIndividual(real_chrom=chromosome)
            population.append(individual)

        return population

    @staticmethod
    def create_known_optima_data(dim: int) -> dict:
        """
        Create individuals at known optima for benchmark functions.

        :param dim: Problem dimension
        :return: Dictionary with optima for different functions
        """
        return {
            "sphere_optimum": GaIndividual(real_chrom=np.zeros(dim)),
            "rastrigin_optimum": GaIndividual(real_chrom=np.zeros(dim)),
            "rosenbrock_optimum": GaIndividual(real_chrom=np.ones(dim)),
            "random_point": GaIndividual(real_chrom=np.random.rand(dim)),
            "boundary_point_lower": GaIndividual(real_chrom=np.full(dim, -5.0)),
            "boundary_point_upper": GaIndividual(real_chrom=np.full(dim, 5.0)),
        }


class TestScenarios:
    """Predefined test scenarios for common testing patterns."""

    @staticmethod
    def optimization_convergence_scenario(
        evaluator, initial_population: List[GaIndividual]
    ) -> dict:
        """
        Create a scenario for testing optimization convergence.

        :param evaluator: Function evaluator
        :param initial_population: Starting population
        :return: Dictionary with scenario data
        """
        # Evaluate initial population
        initial_fitness = []
        for individual in initial_population:
            args = GaEvaluatorArgs(individual)
            fitness = evaluator.evaluate(args)
            initial_fitness.append(fitness)

        if evaluator.extremum() == GaExtremum.MAXIMUM:
            best_initial_idx = np.argmax(initial_fitness)
            worst_initial_idx = np.argmin(initial_fitness)
        else:
            best_initial_idx = np.argmin(initial_fitness)
            worst_initial_idx = np.argmax(initial_fitness)

        return {
            "initial_population": initial_population,
            "initial_fitness": initial_fitness,
            "best_individual": initial_population[best_initial_idx],
            "best_fitness": initial_fitness[best_initial_idx],
            "worst_individual": initial_population[worst_initial_idx],
            "worst_fitness": initial_fitness[worst_initial_idx],
            "population_diversity": np.std(
                [ind.real_chrom for ind in initial_population], axis=0
            ),
            "evaluator": evaluator,
        }

    @staticmethod
    def operator_stress_test_scenario(
        operator, test_population: List[GaIndividual]
    ) -> dict:
        """
        Create a scenario for stress testing operators.

        :param operator: Operator to test
        :param test_population: Population for testing
        :return: Dictionary with scenario data
        """
        return {
            "operator": operator,
            "test_population": test_population,
            "edge_cases": {
                "identical_parents": [test_population[0], test_population[0]],
                "diverse_parents": [test_population[0], test_population[-1]],
                "single_individual": test_population[0],
            },
        }
