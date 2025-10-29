from typing import Callable, List

from autograd import grad
from scipy.stats import gamma
import numpy as np
import sys

from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class WeightedGradientDirectionBasedMutation(GaOperator):
    def __init__(
        self,
        constraint_functions: List[Callable[[np.ndarray], float]],
        p_m: float = 0.1,
        shape_k: int = 2,
        initial_theta: float = 0.5,
    ):
        """Initializes weighted gradient direction-based mutation operator for real-valued
        chromosomes.

        :param constraint_functions: List of constraint functions for mutation, these functions need to use autograd numpy, in another case alhrotim won't work properly
        :type gradient_constraint_functions: List[Callable[[np.ndarray], float]]
        :param p_m: Probability of mutation for weighted gradient direction-based mutation
        :type p_m: float
        """
        self.constraint_functions = constraint_functions
        self.gradient_constraint_functions = [grad(g) for g in constraint_functions]
        self.p_m = p_m
        self.shape_k = shape_k
        self.initial_theta = initial_theta

    def category(self) -> GaOpCategory:
        """Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs weighted gradient direction-based mutation on real-valued chromosomes in the
        population.

        :param args: Container with population and evaluator for
            mutation operation
        :type args: GaOperatorArgs
        :returns: Population with potentially mutated individuals
        :rtype: List[GaIndividual]
        :raises ValueError: If either contraints or evalution method is not workig with autograd. All of these methods needs to be written using autograd.numpy
        """
        population = args.population
        evaulator_wrapper = self.EvaluatorWrapper(args.evaluator)
        gradient_evaluation_function = grad(evaulator_wrapper.evaluate)

        for i in range(len(population)):
            if (
                np.random.uniform(low=0.0, high=1.0 + sys.float_info.epsilon)
                <= self.p_m
            ):
                real_chrom = population[i].real_chrom
                try:
                    gradient_evaluation_function_value = gradient_evaluation_function(
                        real_chrom
                    )
                except Exception as e:
                    raise ValueError(
                        f"""Function is not differentiable with autograd.
Ensure it uses autograd.numpy: {e}"""
                    )

                constraint_function_values = [
                    constraint_function(real_chrom)
                    for constraint_function in self.constraint_functions
                ]
                max_constraint_function_value = max(constraint_function_values)
                try:
                    gradient_constraint_function_values = [
                        grad_func(real_chrom)
                        for grad_func in self.gradient_constraint_functions
                    ]
                except Exception as e:
                    raise ValueError(
                        f"""Function is not differentiable with autograd.
Ensure it uses autograd.numpy: {e}"""
                    )
                lagrange_multipliers = self._find_lagrange_multipliers(
                    gradient_evaluation_function_value,
                    constraint_function_values,
                    gradient_constraint_function_values,
                )
                omegas = []
                for j in range(len(self.constraint_functions)):
                    if constraint_function_values[j] < 0:
                        omega = 0
                    elif constraint_function_values[j] > 0:
                        omega = 1 / np.nextafter(
                            max_constraint_function_value
                            - constraint_function_values[j],
                            np.inf,
                        )
                    else:
                        if lagrange_multipliers[j] is not None:
                            omega = lagrange_multipliers[j]
                        else:
                            omega = 1 / np.nextafter(
                                max_constraint_function_value
                                - constraint_function_values[j],
                                np.inf,
                            )
                    omegas.append(omega)
                beta = gamma.rvs(a=self.shape_k, scale=(self.initial_theta / (j + 1)))
                c = np.array(omegas) @ np.array(gradient_constraint_function_values)
                search_direction_vector = gradient_evaluation_function_value + c
                population[i].real_chrom = (
                    population[i].real_chrom + beta * search_direction_vector
                )
        return population

    class EvaluatorWrapper:
        def __init__(self, evaluator: GaEvaluator):
            """
            Initializes wrapper so it could work properly with autograd
            param evaluator: evaluator used to evaluate individuals

            :param GaEvaluator: Evaluator which is will be used during evaluation
            :type evaluator: GaEvaluator
            """
            self.evaluator = evaluator

        def evaluate(self, real_chrom: np.ndarray):
            """
            Call the orginal method from evaluator and returns its orginal value or its negative value, depending on optimization problem

            :param args: Object representing a particular solution for the
                posited problem.
            :returns: A value representing fitness for this particular
                solution.
            :rtype: float
            """
            individual = GaIndividual()
            individual.real_chrom = real_chrom
            ga_evaluator_args = GaEvaluatorArgs(individual)

            if self.evaluator.extremum == GaExtremum.MAXIMUM:
                result = self.evaluator.evaluate(ga_evaluator_args)
            else:
                result = -self.evaluator.evaluate(ga_evaluator_args)
            return result

    def _find_lagrange_multipliers(
        self,
        gradient_evaluation_function_value: float,
        constraint_function_values: List[float],
        gradient_constraint_function_values: List[float],
    ) -> List[float]:
        """
        Finds Lagrange multipliers for active constraints using gradient information.

        This method solves for Lagrange multipliers in the context of constrained optimization,
        focusing on constraints that are active (i.e., where the constraint function value is zero).
        It constructs a linear system from the gradients of active constraints and attempts to solve
        for the multipliers using least squares.

        :pararm gradient_evaluation_function_value: The gradient value of the evaluation function
                at the current point.
        :param constraint_function_values (List[float]): List of values from each constraint function
                at the current point.
        :type: constraint_function_values: List[float]
        :pararm gradient_constraint_function_values: List of gradient values from each
                constraint function at the current point.
        :type: gradient_constraint_function_values: List[float]
        :returns: List[float]: A list of Lagrange multipliers, where each element corresponds to a constraint.
                Returns None for each inactive constraints if the system is inconsistent/underdetermined.
        :rtype: List[float]
        """
        lagrange_multpliers = []
        constraint_gradients = []
        for i in range(len(constraint_function_values)):
            if constraint_function_values[i] == 0:
                constraint_gradients.append(gradient_constraint_function_values[i])
        if len(constraint_gradients) == 0:
            return len(constraint_function_values) * [None]

        constraint_gradients_array = np.array(constraint_gradients)
        rank_of_gradients = np.linalg.matrix_rank(constraint_gradients_array)
        rank_of_equation = np.linalg.matrix_rank(
            np.column_stack(
                constraint_gradients_array,
                np.full(
                    constraint_gradients_array.shape[0],
                    gradient_evaluation_function_value,
                ),
            )
        )
        if rank_of_gradients != rank_of_equation:
            return len(constraint_function_values) * [None]
        solution_index = 0
        solutions = np.linalg.lstsq(constraint_gradients_array)
        for i in range(len(constraint_function_values)):
            if constraint_function_values[i] == 0:
                lagrange_multpliers.append(solutions[i])
                solution_index += 1
            else:
                lagrange_multpliers.append(None)
