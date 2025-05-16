from core.Population import Population
from core.Representation import Representation
from core.operators.Crossover import Crossover


class KingStrategyCrossover(Crossover):
    """
    A class that implements Multi Parent Feature Wise Crossover.
    The population passed to a cross method is expected to have all the individual values calculated.
    Internally it is used to select the first best one individual and perform crossing with it.

    :param how_many_individuals: The number of individuals to create in the offspring. It is ignored, as this crossover always creates the same size of the offspring as the population_parent size.
    :param probability: The probability of performing the crossover operation.
    :param minimum_is_best: If True, the individual with the smallest value is considered the best, otherwise the largest value is considered the best.
    :param crossover_function: Any crossover function that supports ``Representation.REAL``, that is not a KingStrategyCrossover,
        which can work on Population of 2 Individuals and which returns a Population of 1 Individual.

    allowed_representation: [Representation.REAL]
    """

    allowed_representation = [Representation.REAL]

    def __init__(
            self,
            how_many_individuals: int,
            probability: float = 0,
            minimum_is_best: bool = True,
            crossover_function: Crossover = None
    ):
        super().__init__(how_many_individuals, probability)
        self.minimum_is_best: bool = minimum_is_best
        self.crossover_function: Crossover = crossover_function

    def _cross(self, population_parent: Population) -> Population:
        if self.minimum_is_best:
            best_individual_index = min(range(population_parent.population_size), key=lambda i: population_parent.population[i].value)
        else:
            best_individual_index = max(range(population_parent.population_size), key=lambda i: population_parent.population[i].value)

        children = Population(population=[])
        for j in range(population_parent.population_size):
            if j == best_individual_index:
                continue
            tmp_population = Population(
                population=[
                    population_parent.population[best_individual_index],
                    population_parent.population[j]
                ]
            )

            children.add_to_population(self.crossover_function.cross(tmp_population))

        return children
