from typing import List
import copy
import sys

import numpy as np

from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class SimulatedAnnealingBasedMuttaion1(GaOperator):
    def __init__(self, Q_parameter: int = 10):
        """
        Initializes SimulatedAnnealingBasedMuttaion version 1
        operator for real-valued chromosomes.

        :param Q_parameter: parameter which determines after how many
            iteration should the cooling procedure related to annealing.
        :type p_um: int
        """
        self.Q_parameter = Q_parameter

    def category(self):
        """
        Returns the category of the operator, used to classify its
        type in the evolutionary algorithm framework.

        :returns: The operator category indicating real-valued mutation
        """
        return GaOpCategory.REAL_MUTATION

    def perform(self, args: GaOperatorArgs) -> List[GaIndividual]:
        """
        Performs annealing based muttaion on real-valued chromosomes
        in the population.

        :param args: Container with population and evaluator for
            mutation operation
        :type args: GaOperatorArgs
        :returns: Population with potentially mutated individuals
        :rtype: List[GaIndividual]
        """
        population = args.population
        counter = 1
        stagnation_count = 1
        population_size = len(population)
        is_maximilation_problem = args.evaluator.extremum == GaExtremum.MAXIMUM
        for individual in population:
            ga_evaluators_args = GaEvaluatorArgs(individual)
            individual.value = args.evaluator.evaluate(ga_evaluators_args)

        sorted_population = sorted(
            population, key=lambda ind: ind.value, reverse=is_maximilation_problem
        )
        worst_individual = sorted_population[population_size - 1]
        best_individual = sorted_population[0]
        if worst_individual.value <= 0:
            t_0 = 1 / (-worst_individual.value + sys.float_info.epsilon)
        else:
            t_0 = 1 / worst_individual.value

        domains = args.evaluator.real_domain()
        for i in range(population_size):
            individual = population[i]
            random_indvidual = copy.deepcopy(individual)
            random_indvidual.real_chrom = np.array(
                [np.random.uniform(low, high) for low, high in domains]
            )
            ga_evaluators_args = GaEvaluatorArgs(random_indvidual)
            random_indvidual.value = args.evaluator.evaluate(ga_evaluators_args)
            if (
                is_maximilation_problem and individual.value < random_indvidual.value
            ) or (
                not is_maximilation_problem
                and individual.value > random_indvidual.value
            ):
                population[i] = random_indvidual
            else:
                p_accept = counter ** (
                    (1 / individual.value - 1 / random_indvidual.value) / t_0
                )
                if p_accept < np.random.uniform(
                    low=np.nextafter(0, 1), high=np.nextafter(1, 0)
                ):
                    population[i] = random_indvidual

            if (is_maximilation_problem and best_individual.value < population[i]) or (
                not is_maximilation_problem
                and best_individual.value > population[i].value
            ):
                best_individual = population[i]
            else:
                stagnation_count += 1

            if stagnation_count >= self.Q_parameter:
                counter = 1
                sorted_population = sorted(
                    population,
                    key=lambda ind: ind.value,
                    reverse=is_maximilation_problem,
                )
                worst_individual = sorted_population[population_size - 1]
                if worst_individual.value <= 0:
                    t_0 = 1 / (-worst_individual.value + sys.float_info.epsilon)
                else:
                    t_0 = 1 / worst_individual.value
            else:
                counter += 1
        return population
