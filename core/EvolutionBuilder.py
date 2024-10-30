from typing import Type, Callable, List, Tuple, Self
from selection import Selection
from crossover import Crossover
from mutation import Mutation


class EvolutionBuilder:
    """
    A builder class for constructing an instance of the Evolution class. This class follows the
    builder pattern to ensure that all required parameters are properly set before the
    Evolution object is created.

    :param evolution_reference: A reference to the Evolution class, not an instance of the class.
    """

    def __init__(self, evolution_reference: Evolution = Evolution):
        """
        Initialize the EvolutionBuilder with a reference to the Evolution class.

        :param evolution_reference: A reference to the Evolution class (not an instance).
        """
        self.evolution_reference = evolution_reference
        self.selection = None
        self.crossover = None
        self.mutation = None
        self.elitism = None
        self.fitness_function = None
        self.population_generator = None
        self.events = []
        self.jobs = []
        self.terminator = None
        self.clamp_strategy = None
        self.population_size = None
        self.individual_size = None
        self.max_epoch = None
        self.maximize = None
        self.representation = None

    def _validate(self, var, var_type, var_name):
        """
        Validate that a variable is of the expected type.

        :param var: The variable to validate.
        :param var_type: The expected type of the variable.
        :param var_name: The name of the variable (for error reporting).
        :raises ValueError: If the variable is not of the expected type.
        """
        if not isinstance(var, var_type):
            raise ValueError(f"{var_name} must be of type {var_type.__name__}")

    def set_selection(self, selection: Selection) -> Self:
        """
        Set the selection operator for the evolution process.

        :param selection: The selection operator to use.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(selection, Selection, "Selection")
        self.selection = selection
        return self

    def set_crossover(self, crossover: Crossover) -> Self:
        """
        Set the crossover operator for the evolution process.

        :param crossover: The crossover operator to use.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(crossover, Crossover, "Crossover")
        self.crossover = crossover
        return self

    def set_mutation(self, mutation: Mutation) -> Self:
        """
        Set the mutation operator for the evolution process.

        :param mutation: The mutation operator to use.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(mutation, Mutation, "Mutation")
        self.mutation = mutation
        return self

    def use_preset(self, preset: OperatorsPreset) -> Self:
        """
        Set the selection, crossover, and mutation operators using a preset.

        :param preset: An OperatorsPreset instance with predefined operators.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self.set_selection(preset.selection)
        self.set_crossover(preset.crossover)
        self.set_mutation(preset.mutation)
        return self

    def set_elitims(self, elitims: Elitism) -> Self:
        """
        Set the elitism strategy for the evolution process.

        :param elitims: The elitism strategy to use.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(elitims, Elitism, "Elitism")
        self.elitism = elitims
        return self

    def set_fitness_function(self, fitness_function: FitnessFunction) -> Self:
        """
        Set the fitness function for the evolution process.

        :param fitness_function: The fitness function to use.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(fitness_function, FitnessFunction, "Fitness Function")
        self.fitness_function = fitness_function
        return self

    def set_population_generator(self, population_function: Callable[[int, int], List[Tuple[(int, int)]]],
                                 population_size: int|None = None,
                                 individual_size: int|None = None,
                                 variable_domains: List[Tuple[(int, int)]]|None = None) -> Self:
        """
        Set the population generator for the evolution process.

        :param population_function: The function responsible for generating the population.
        :param population_size: The size of the population (optional).
        :param individual_size: The size of an individual (optional).
        :param variable_domains: The variable domains for generating individuals (optional).
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(population_function, Callable, "Population Generator")
        self.population_generator = population_function
        self.population_size = population_size
        self.individual_size = individual_size
        self.variable_domains = variable_domains
        return self

    def set_population_size(self, population_size: int) -> Self:
        """
        Set the population size for the evolution process.

        :param population_size: The number of individuals in the population.
        :return: The EvolutionBuilder instance, allowing for method chaining.
        """
        self._validate(population_size, int, "Population Size")
        self.population_size = population_size
        return self