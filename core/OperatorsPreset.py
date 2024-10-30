from dataclasses import dataclass
from core.Mutation import Mutation
from core.Crossover import Crossover
from core.Selection import Selection


@dataclass
class OperatorsPreset:
    selection: Selection|None = None
    crossover: Crossover|None = None
    mutation: Mutation|None = None
