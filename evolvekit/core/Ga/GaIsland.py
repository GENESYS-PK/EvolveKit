import heapq
import copy
import random
from typing import List

import numpy as np

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.clamp_strategy.ClampStrategyFactory import ClampStrategyFactory
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.GaResults import GaResults
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.enums.GaAction import GaAction
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs


class GaIsland(GaState):
    inspector: GaInspector
    selection: GaOperator
    real_crossover: GaOperator | None
    real_mutation: GaOperator | None
    bin_crossover: GaOperator | None
    bin_mutation: GaOperator | None

    def __verify(self):
        if self.evaluator.bin_length() == 0 and len(self.evaluator.real_domain()) == 0:
            raise ValueError(
                "Both bin_length is 0 and real_domain is empty list in evaluator."
            )

    def __initialize(self):
        np.random.seed(self.seed)
        self.inspector.initialize()
        self.statistic_engine.start()
        self.selection.initialize(self)
        if self.real_crossover:
            self.real_crossover.initialize(self)
        if self.real_mutation:
            self.real_mutation.initialize(self)
        if self.bin_crossover:
            self.bin_crossover.initialize(self)
        if self.bin_mutation:
            self.bin_mutation.initialize(self)

    def __evaluate(self):
        for indiv in self.current_population:
            self.evaluator.evaluate(GaEvaluatorArgs(indiv))

    def __evolve(self):
        if self.evaluator.extremum() == GaExtremum.MAXIMUM:
            elite_population = heapq.nlargest(
                self.elite_size, self.current_population, key=lambda indiv: indiv.value
            )
        else:
            elite_population = heapq.nsmallest(
                self.elite_size, self.current_population, key=lambda indiv: indiv.value
            )

        self.elite_population = copy.deepcopy(elite_population)
        self.selected_population = copy.deepcopy(
            self.selection.perform(GaOperatorArgs(self, self.selection.category()))
        )
        self.offspring_population = copy.deepcopy(self.current_population)

        if self.real_crossover:
            real_crossover_list = self.__perform_crossover(self.real_crossover)
            for offspring, crossover_indiv in zip(
                self.offspring_population, real_crossover_list
            ):
                offspring.real_chrom = crossover_indiv.real_chrom

        if self.bin_crossover:
            bin_crossover_list = self.__perform_crossover(self.real_crossover)
            for offspring, crossover_indiv in zip(
                self.offspring_population, bin_crossover_list
            ):
                offspring.bin_chrom = crossover_indiv.bin_chrom

        temp_offspring = []
        if self.real_mutation:
            temp_offspring = self.real_mutation.perform(
                GaOperatorArgs(self, self.selection.category())
            )

        if self.bin_mutation:
            temp_offspring = self.bin_mutation.perform(
                GaOperatorArgs(self, self.selection.category())
            )

        indices = np.random.choice(
            len(self.offspring_population),
            size=len(self.elite_population),
            replace=False,
        )

        for i in range(len(self.offspring_population)):
            if np.random.random() < self.mutation_prob:
                self.offspring_population[i] = temp_offspring[i]

        if self.real_clamp_strategy != GaClampStrategy.NONE:
            for indiv in self.offspring_population:
                for i in range(len(self.evaluator.real_domain())):
                    gene_value = indiv.real_chrom[i]
                    domain = self.evaluator.real_domain()[i]
                    lower, upper = domain
                    if not lower <= gene_value <= upper:
                        indiv.real_chrom[i] = ClampStrategyFactory.get_clamp_strategy(
                            self.real_clamp_strategy
                        )(gene_value, domain)

        for i, idx in enumerate(indices):
            self.offspring_population[i] = self.elite_population[idx]

        self.current_population = self.offspring_population
        self.selected_population = []
        self.offspring_population = []
        self.elite_population = []

    def __perform_crossover(self, crossover: GaOperator) -> List[GaIndividual]:
        crossover_list = []
        while len(crossover_list) < self.population_size:
            if np.random.random() < self.crossover_prob:
                crossover_list.extend(
                    crossover.perform(GaOperatorArgs(self, crossover.category()))
                )
            else:
                crossover_list.append(random.choice(self.selected_population))
        crossover_list = crossover_list[: self.population_size - 1]
        return crossover_list

    def __finish(self) -> GaResults:
        pass

    def run(self) -> GaResults:
        self.__verify()
        self.__initialize()

        while True:
            self.__evaluate()
            self.statistic_engine.advance()
            action = self.inspector.inspect(self.statistic_engine)
            if action is GaAction.TERMINATE:
                break

        return self.__finish()

    def set_elite_count(self, count: int):
        self.elite_size = count

    def set_crossover_probability(self, prob: float):
        self.crossover_prob = prob

    def set_mutation_probability(self, prob: float):
        self.mutation_prob = prob

    def set_max_generations(self, count: int):
        self.max_generations = count

    def set_seed(self, seed: int):
        self.seed = seed

    def set_evaluator(self, evaluator: GaEvaluator):
        self.evaluator = evaluator

    def set_inspector(self, inspector: GaInspector):
        self.inspector = inspector

    def set_operator(self, operator: GaOperator):
        if operator.category() == GaOpCategory.SELECTION:
            self.selection = operator
        elif operator.category() == GaOpCategory.REAL_CROSSOVER:
            self.real_crossover = operator
        elif operator.category() == GaOpCategory.REAL_MUTATION:
            self.real_mutation = operator
        elif operator.category() == GaOpCategory.BIN_CROSSOVER:
            self.bin_crossover = operator
        elif operator.category() == GaOpCategory.BIN_MUTATION:
            self.bin_mutation = operator

    def set_real_clamp_strategy(self, strategy: GaClampStrategy):
        self.real_clamp_strategy = strategy

    def set_population_size(self, size: int):
        self.population_size = size
