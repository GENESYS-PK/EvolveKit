from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from core import Population
from core.evolution import EvolutionState
from core.operators import Mutation
from core.Individual import Individual
from core.Representation import Representation


class NonlinearMutation1(Mutation):
    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float, domains: NDArray[np.floating]):
        """
        Constructor for the NonlinearMutation1 class.

        NOTE: All individuals need to have the same length of all chromosomes, and the same number of chromosomes, but it is not validated.

        Parameters:
            probability (float): Probability of mutation (between 0 and 1).
            domains (NDArray[np.floating]): A numpy array of shape (n, 2) where domains[i][0] is x_min and domains[i][1] is x_max for a chromosome value.
               It is inclusive, so the mutated chromosomes values are within [x_min, x_max].
               Each pair should describe the range of values for a specific chromosome that are allowed by the considered optimization problem.
               It should not be a range that is narrower than the domain allowed by the considered optimization problem.
               It is not desired to use this parameter to achieve narrower clamping of mutated values.
        """
        super().__init__(probability)
        self.domains: NDArray[np.floating] = domains
        self.__parent_population_copy: Population = None


    def _init_mutation_round(self, evolution_state: EvolutionState):
        # Save the population before selection and crossover steps as a copy.
        # This is done to prevent creating mutated individuals based on other individuals mutated in the same round
        # This is how this mutation is defined to work. 
        # It is defined to mutate individuals based on population which is before crossover and selection step.
        self.__parent_population_copy = deepcopy(evolution_state.current_population)


    def _mutate(self, individual: Individual, population: Population) -> None:
        """
        Parameters:
            individual (Individual): A single individual to mutate.
            population (Population): The population containing individuals.

        Returns:
            None: The population with mutated individuals.
        """


        Xq: Individual = np.random.random_choice(self.__parent_population_copy.population)
        k = np.random.randint(0, len(individual.chromosome))

        range1 = (self.domains[k][0], min(individual.chromosome[k], Xq.chromosome[k]))
        range2 = (max(individual.chromosome[k], Xq.chromosome[k]), self.domains[k][1])

        range1_length = range1[1] - range1[0]
        range2_length = range2[1] - range2[0]

        sum_length = range1_length + range2_length

        if np.random.uniform(0, 1) <= range1_length / sum_length:
            individual.chromosome[k] = np.random.uniform(*range1)
        else:
            individual.chromosome[k] = np.random.uniform(*range2)


