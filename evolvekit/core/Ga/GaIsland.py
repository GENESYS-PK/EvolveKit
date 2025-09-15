import heapq
import copy
from typing import List

import numpy as np

from evolvekit.core.Ga.GaEvaluatorArgs import GaEvaluatorArgs
from evolvekit.core.Ga.GaIndividual import GaIndividual
from evolvekit.core.Ga.helpers.ClampStrategy import get_clamp_strategy
from evolvekit.core.Ga.GaEvaluator import GaEvaluator
from evolvekit.core.Ga.GaInspector import GaInspector
from evolvekit.core.Ga.GaResults import GaResults
from evolvekit.core.Ga.GaState import GaState
from evolvekit.core.Ga.enums.GaAction import GaAction
from evolvekit.core.Ga.enums.GaExtremum import GaExtremum
from evolvekit.core.Ga.enums.GaOpCategory import GaOpCategory
from evolvekit.core.Ga.enums.GaClampStrategy import GaClampStrategy
from evolvekit.core.Ga.helpers.GaGenerateRandomPopulation import (
    generate_random_population,
)
from evolvekit.core.Ga.operators.GaOperator import GaOperator
from evolvekit.core.Ga.operators.GaOperatorArgs import GaOperatorArgs
from evolvekit.operators.Ga.selection.RankSelection import RankSelection
from evolvekit.operators.Ga.crossover.real.OnePointCrossover import (
    OnePointCrossover as OnePointCrossover,
)
from evolvekit.operators.Ga.mutation.real.VirusInfectionMutation import (
    VirusInfectionMutation as VirusInfectionMutation,
)
from evolvekit.operators.Ga.crossover.binary.OnePointCrossover import (
    OnePointCrossover as BinaryOnePointCrossover,
)
from evolvekit.operators.Ga.mutation.binary.VirusInfectionMutation import (
    VirusInfectionMutation as BinaryVirusInfectionMutation,
)


class GaIsland(GaState):
    inspector: GaInspector | None
    selection: GaOperator | None
    real_crossover: GaOperator | None
    real_mutation: GaOperator | None
    bin_crossover: GaOperator | None
    bin_mutation: GaOperator | None

    def __init__(self):
        super().__init__()
        self.population_size = 100
        self.elite_size = 0
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.max_generations = 200
        self.seed = 0
        self.real_clamp_strategy = GaClampStrategy.NONE

        # FIXME: I know I shouldn't do this in a constructor. It's a temporary solution until more basic operators will be implemented.
        dim = 15
        virus_vectors = [
            [0.0] * dim,  # full reset pattern
            [None if i < dim // 2 else 0.0 for i in range(dim)],  # half reset
            [
                np.random.normal(0.0, 0.7) if np.random.rand() < 0.5 else None
                for _ in range(dim)
            ],
        ]

        virus_vectors_binary = [
            [0] * dim,  # full reset pattern
            ["*" if i < dim // 2 else 0 for i in range(dim)],  # half reset
            [
                np.random.choice([0, 1]) if np.random.rand() < 0.5 else "*"
                for _ in range(dim)
            ],
        ]

        self.inspector = None
        self.selection = RankSelection(target_population=15)
        self.real_crossover = OnePointCrossover()
        self.real_mutation = VirusInfectionMutation(virus_vectors=virus_vectors)
        self.bin_crossover = BinaryOnePointCrossover()
        self.bin_mutation = BinaryVirusInfectionMutation(
            virus_vectors=virus_vectors_binary
        )
        self.__binary_representation = False
        self.__real_representation = False

    def __verify(self):
        if not self.evaluator:
            raise TypeError("Evaluator cannot be empty")

        if self.evaluator.bin_length() == 0 and not self.evaluator.real_domain():
            raise ValueError(
                "Both bin_length is 0 and real_domain is empty list in evaluator."
            )
        if self.evaluator.bin_length() > 0:
            self.__binary_representation = True
            if not self.bin_mutation or not self.bin_crossover:
                raise ValueError(
                    "Binary chromosome set as active but binary mutation or binary crossover is empty."
                )

        if not self.selection:
            raise TypeError("Selection cannot be empty")

        if self.evaluator.real_domain():
            self.__real_representation = True
            if not self.real_mutation or not self.real_crossover:
                raise ValueError(
                    "Real chromosome set as active but real mutation or real crossover is empty."
                )

        if (
            self.real_crossover
            and self.real_crossover.category() != GaOpCategory.REAL_CROSSOVER
        ):
            raise TypeError(
                f"Expected type for real crossover is REAL_CROSSOVER, got: {self.real_crossover.category().name}."
            )

        if (
            self.real_mutation
            and self.real_mutation.category() != GaOpCategory.REAL_MUTATION
        ):
            raise TypeError(
                f"Expected type for real mutation is REAL_MUTATION, got: {self.real_mutation.category().name}."
            )

        if (
            self.bin_crossover
            and self.bin_crossover.category() != GaOpCategory.BIN_CROSSOVER
        ):
            raise TypeError(
                f"Expected type for binary crossover is BIN_CROSSOVER, got: {self.bin_crossover.category().name}."
            )

        if (
            self.bin_mutation
            and self.bin_mutation.category() != GaOpCategory.BIN_MUTATION
        ):
            raise TypeError(
                f"Expected type for binary mutation is BIN_MUTATION, got: {self.bin_mutation.category().name}."
            )

        if self.population_size <= 0:
            raise ValueError("Population size must be greater than 0.")

        if self.elite_size < 0:
            raise ValueError("Elite size must be greater than or equal 0.")

        if not 0 <= self.crossover_prob <= 1:
            raise ValueError("Crossover probability must be between 0 and 1.")

        if not 0 <= self.mutation_prob <= 1:
            raise ValueError("Mutation probability must be between 0 and 1.")

        if self.max_generations <= 0:
            raise ValueError("Max generations must be greater than 0.")

    def __initialize(self):
        np.random.seed(self.seed)
        self.current_population = generate_random_population(
            self.evaluator, self.population_size
        )
        if self.inspector:
            self.inspector.initialize()
        self.statistic_engine.start(self)
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
            indiv.value = self.evaluator.evaluate(GaEvaluatorArgs(indiv))

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
        self.offspring_population = [
            GaIndividual(
                np.array([], dtype=np.float64), np.array([], dtype=np.uint8), 0
            )
            for _ in range(self.population_size)
        ]

        if self.__binary_representation:
            bin_crossover_list = self.__perform_crossover(self.bin_crossover)
            for offspring, crossover_indiv in zip(
                self.offspring_population, bin_crossover_list
            ):
                offspring.bin_chrom = crossover_indiv.bin_chrom

            mutation_offspring = self.bin_mutation.perform(
                GaOperatorArgs(self, self.bin_mutation.category())
            )
            self.__assignPopulationAfterMutation(mutation_offspring)

        if self.__real_representation:
            real_crossover_list = self.__perform_crossover(self.real_crossover)
            for offspring, crossover_indiv in zip(
                self.offspring_population, real_crossover_list
            ):
                offspring.real_chrom = crossover_indiv.real_chrom

            mutation_offspring = self.real_mutation.perform(
                GaOperatorArgs(self, self.real_mutation.category())
            )
            self.__assignPopulationAfterMutation(mutation_offspring)

        if self.real_clamp_strategy != GaClampStrategy.NONE:
            for indiv in self.offspring_population:
                domain_count = len(self.evaluator.real_domain())
                for i in range(domain_count):
                    gene_value = indiv.real_chrom[i]
                    domain = self.evaluator.real_domain()[i]
                    lower, upper = domain
                    if not lower <= gene_value <= upper:
                        indiv.real_chrom[i] = get_clamp_strategy(
                            self.real_clamp_strategy
                        )(gene_value, domain)

        indices = np.random.choice(
            len(self.offspring_population),
            size=len(self.elite_population),
            replace=False,
        )

        for elite_index, offspring_index in enumerate(indices):
            self.offspring_population[offspring_index] = self.elite_population[
                elite_index
            ]

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
                crossover_list.append(np.random.choice(self.selected_population))
        crossover_list = crossover_list[: self.population_size]
        return crossover_list

    def __assignPopulationAfterMutation(self, mutation_offspring: List[GaIndividual]):
        for i in range(self.population_size):
            if np.random.random() < self.mutation_prob:
                self.offspring_population[i] = mutation_offspring[i]

    def __finish(self) -> GaResults:
        if self.inspector:
            self.inspector.finish(self.statistic_engine)
        return GaResults(self.statistic_engine)

    def run(self) -> GaResults:
        self.__verify()
        self.__initialize()

        while True:
            self.__evaluate()
            self.statistic_engine.advance(self)
            if self.inspector:
                action = self.inspector.inspect(self.statistic_engine)
                if action is GaAction.TERMINATE:
                    break
            if self.statistic_engine.generation > self.max_generations:
                break
            self.__evolve()

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
        match operator.category():
            case GaOpCategory.SELECTION:
                self.selection = operator
            case GaOpCategory.REAL_CROSSOVER:
                self.real_crossover = operator
            case GaOpCategory.REAL_MUTATION:
                self.real_mutation = operator
            case GaOpCategory.BIN_CROSSOVER:
                self.bin_crossover = operator
            case GaOpCategory.BIN_MUTATION:
                self.bin_mutation = operator

    def set_real_clamp_strategy(self, strategy: GaClampStrategy):
        self.real_clamp_strategy = strategy

    def set_population_size(self, size: int):
        self.population_size = size
