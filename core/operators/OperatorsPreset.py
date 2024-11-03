from dataclasses import dataclass
from core.operators.Mutation import Mutation
from core.operators.Crossover import Crossover
from core.operators.Selection import Selection


@dataclass
class OperatorsPreset:
    selection: Selection | None = None
    crossover: Crossover | None = None
    mutation: Mutation | None = None
