from copy import deepcopy
import numpy as np
from core import Population
from core.evolution import EvolutionState
from core.operators import Mutation
from core.Individual import Individual
from core.Representation import Representation


class SelectionFollowerMutation(Mutation):
    allowed_representation = [Representation.REAL]

    def __init__(self, probability: float, gamma: float):
        """
        Constructor for the SelectionFollowerMutation class.

        NOTE: All individuals need to have the same length of all chromosomes, and the same number of chromosomes, but it is not validated.

        Parameters:
            probability (float): Probability of mutation (between 0 and 1).
            gamma (float): Scaling factor, must be in range <0; 1), where 0 means no mutation.

        Throws:
            ValueError: If gamma is not in the range <0; 1).
        """
        super().__init__(probability)

        if not (0 < gamma <= 1):
            raise ValueError("Gamma must be in range <0; 1)")
        self.gamma = gamma

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

        # Save time from doing useless calculations when gamma is 0, which effectively means no mutation will take place ever
        if self.gamma == 0:
            return
        
        # determine the index of the individual in the population,
        # to get the corresponding individual from population from before selection and crossover steps
        # to compare against 2 randomly selected individuals from the population
        j = population.population.index(individual)

        # Get two random individuals from the parent population that are different from individual
        while True:
            y_individual: Individual = np.random.choice(self.__parent_population_copy.population)
            z_individual: Individual = np.random.choice(self.__parent_population_copy.population)
            # If y and z are the same, this will be equivalent to mutation with gamma = 0, so no mutation will take place
            # It is allowed scenario for this mutation
            # For small populations, it will be much less likely for mutation to ever take place, which is to be expected
            
            # Check if the drawn individuals are different from the current individual

            if y_individual is not self.__parent_population_copy.population[j] and z_individual is not self.__parent_population_copy.population[j]:
                break
        
        # All individuals need to have the same length, but it is not validated
        for i in range(len(individual.chromosome)):
            k = np.random.uniform(0, self.gamma)
            individual.chromosome[i] = self.__parent_population_copy.population[j].chromosome[i] + k * (z_individual.chromosome[i]- y_individual.chromosome[i])

        # NOTE: If RND is greater than set self.probability, this mutation defines, that instead of the above modifications
        # the individual from passed population should be replaced with an individual at the same index j from self.__parent_population_copy.
        # But since this protected method is only called after the probability check, it is impossible to do it here.
        # To do so, public method mutate() must be overridden.

