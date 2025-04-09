import numpy as np
from collections import defaultdict
from core import Population
from core.Individual import Individual
from core.operators import Crossover
from core.Representation import Representation


class EGAXMultiUniformCrossover(Crossover):
    """
    Operator krzyżowania wieloosobniczego równomiernego zgodny z metodą EGAX.
    
    Chromosomy posiadają dodatkowy znacznik (np. ostatni gen) reprezentujący operator krzyżowania.
    Rodzice są grupowani wg znacznika, a krzyżowanie zachodzi osobno w każdej grupie.
    
    allowed_representation: [Representation.REAL]
    """
    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 1.0):
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        parents = population_parent.population

        groups = defaultdict(list)
        for individual in parents:
            marker = individual.chromosome[-1]
            groups[marker].append(individual)

        offspring = []

        for group in groups.values():
            group_size = len(group)
            if group_size < 2:
                continue  

            for _ in range(group_size):
                new_chromosome = []
                num_genes = len(group[0].chromosome)
                for gene_idx in range(num_genes):
                    gene_values = [ind.chromosome[gene_idx] for ind in group]
                    selected_value = np.random.choice(gene_values)
                    new_chromosome.append(selected_value)

                child = Individual(chromosome=np.array(new_chromosome, dtype=float), value=0)
                offspring.append(child)

        return Population(population=offspring)
