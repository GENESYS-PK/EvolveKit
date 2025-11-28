from evolvekit.benchmarks.SphereEvaluator import SphereEvaluator
from evolvekit.benchmarks.RastriginEvaluator import RastriginEvaluator
from evolvekit.benchmarks.RosenbrockEvaluator import RosenbrockEvaluator


def evaluator_factory(evaluator_type, dimension):
    """
    Create an evaluator of specified type and dimension.

    :param evaluator_type: Type of evaluator ("sphere", "rastrigin", "rosenbrock")
    :param dimension: Number of dimensions for the evaluation function
    :return: Evaluator instance
    :raises ValueError: If evaluator_type is not recognized
    """
    evaluator_map = {
        "sphere": SphereEvaluator,
        "rastrigin": RastriginEvaluator,
        "rosenbrock": RosenbrockEvaluator,
    }

    if evaluator_type not in evaluator_map:
        available_types = ", ".join(evaluator_map.keys())
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}. Available types: {available_types}"
        )

    return evaluator_map[evaluator_type](dim=dimension)


def all_evaluators_factory(dimension):
    """
    Create all benchmark evaluators for specified dimension.

    :param dimension: Number of dimensions for the evaluation functions
    :return: Dictionary mapping evaluator names to evaluator instances
    """
    return {
        "sphere": SphereEvaluator(dim=dimension),
        "rastrigin": RastriginEvaluator(dim=dimension),
        "rosenbrock": RosenbrockEvaluator(dim=dimension),
    }
