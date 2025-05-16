import numpy as np
from collections import defaultdict
from core import Population
from core.Individual import Individual
from core.operators import Crossover
from core.Representation import Representation


class ExtendedGACrossover(Crossover):
    """
    Multi-parent uniform crossover operator compatible with the EGAX method.
    
    Chromosomes contain an additional marker (e.g., the last gene) representing the crossover operator.
    Parents are grouped based on this marker, and crossover is performed separately within each group.
    
    allowed_representation: [Representation.REAL]
    """
    allowed_representation = [Representation.REAL]

    def __init__(self, how_many_individuals: int, probability: float = 1.0):
        super().__init__(how_many_individuals, probability)

    def _cross(self, population_parent: Population) -> Population:
        parents = population_parent.population

        #zakładam że znacznik alfa to ostatni gen w chromosomie by ułatwić, możemy przegadać jak to inaczej zrobić.
        groups = defaultdict(list)
        for individual in parents:
            marker = individual.chromosome[-1]
            groups[marker].append(individual)

        offspring = []

        for group in groups.values():
            group_size = len(group)
            if group_size < 2:
                continue  # aby nie krzyżować 1-osobowych grupek

        # zrobilem tutaj swoje krzyzowanie bo nikt nie zaimplementował u nas tego

            chromosomes = [ind.chromosome for ind in group]
            num_genes = len(chromosomes[0])

            for _ in range(len(group)):
                child_genes = [
                    np.random.choice([chrom[gene_idx] for chrom in chromosomes])
                    for gene_idx in range(num_genes)
                ]
                child = Individual(chromosome=np.array(child_genes, dtype=float), value=0)
                offspring.append(child)

        return Population(population=offspring)
