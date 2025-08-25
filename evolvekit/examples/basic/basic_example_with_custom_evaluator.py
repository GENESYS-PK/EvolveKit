import numpy as np

from evolvekit.core.Ga.GaIsland import GaIsland
from typing import List, Tuple

from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum


def basic_example_with_custom_evaluator() -> None:
    """
    This is basic example to show how to use our library with your own custom evaluator.

    To create your own optimization problem, you must create a class that inherits from GaEvaluator and implements
    the following methods:

    evaluate(args: GaEvaluatorArgs): float – the main evaluation method in which you define your optimization problem.

    extremum(): GaExtremum – a method that returns a value depending on whether the optimization problem being defined
    is a minimization or maximization problem.

    In addition, you should implement at least one of the following methods, depending on whether you will be
    operating on the real representation, the binary representation, or both.

    real_domain(): list[tuple[float, float]] – this method returns a list of ranges (min, max) of individual genes
    in the real chromosome.

    bin_length(): int – this method returns the number of bits (genes) in a binary chromosome.

    Example of simple evaluator for quadratic problem.
    """

    class QuadraticEvaluator(GaEvaluator):
        """
        Simple quadratic function:
        f(x) = 2x^2 + 3x + 5
        Global minimum at x = -0.75
        Default domain: [-10.0, 10.0]
        """

        def __init__(self, bounds: Tuple[float, float] = (-10.0, 10.0)):
            """
            Initialize the quadratic evaluator.

            :param bounds: (lower, upper) bound for x
            :raises ValueError: if lower >= upper
            """
            if bounds[0] >= bounds[1]:
                raise ValueError("Invalid bounds: lower must be < upper")
            self._bounds = bounds
            self._dim = 15  # length of chromosome

        def evaluate(self, args: GaEvaluatorArgs) -> float:
            """
            Compute fitness value for the provided chromosome.

            :param args: GaEvaluatorArgs containing real_chrom
            :returns: function value f(x)
            """
            x = args.real_chrom[: self._dim]
            val = 2.0 * x**2 + 3.0 * x + 5.0
            return float(np.sum(val))

        def extremum(self) -> GaExtremum:
            """
            Returns the optimization direction.

            :returns: GaExtremum.MINIMUM
            """
            return GaExtremum.MINIMUM

        """
        The following method is implemented because we want to operate on the real representation
        """

        def real_domain(self) -> List[Tuple[float, float]]:
            """
            Return domain for the single variable.

            :returns: list of (lower, upper) tuple
            """
            return [self._bounds] * self._dim

        """
        If you want to operate on binary representation, it is enough that you implement this method,
        for example, in this way:
        
        def bin_length(self) -> int:
        # Length of the *binary* chromosome in bits.
            return 5
            
        """

    """"
    That's all!!! Now you can use your own Evaluator and optimizing your problem. 
    """
    evaluator = QuadraticEvaluator()

    ga = GaIsland()
    ga.set_evaluator(evaluator)

    results = ga.run()
    logged_generations = results.total_generations

    print("Quadratic optimization")
    print(f"  best fitness: {results.value:.6f}")
    print(f"  generations:  {logged_generations}")
