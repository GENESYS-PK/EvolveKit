from dataclasses import dataclass
from Mutation import Mutation
from Crossover import Crossover
from Selection import Selection


@dataclass
class OperatorsPreset:
    selection: Selection = Selection(1,True)
    crossover: Crossover = Crossover(1)
    mutation: Mutation = Mutation()

