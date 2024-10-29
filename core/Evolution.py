from Mutation import Mutation
from Selection import Selection
from Crossover import Crossover
from Elitism import Elitism
from FitnessFunction import FitnessFunction
from Job import Job
from Population import Population
from Expression import Expression
from EvolutionState import EvolutionState
from EventListenerType import EventListenerType
from OperatorsPreset import OperatorsPreset


from typing import List, Tuple, Callable, Self


class Evolution:
    def __init__(
        self,
        mutation: Mutation = None,
        selection: Selection= None,
        crossover: Crossover= None,
        elitism: Elitism= None,
        fitness_function: FitnessFunction= None,
        job_queue: List[Job]= None,
        init_population: Population= None,
        population_size: int= None,
        terminator: Expression= None,
        maximize: bool= None,
        events: Tuple[List[Callable[[EvolutionState], None]], ...]= None,
    ):
        self.mutation = mutation
        self.selection = selection
        self.crossover = crossover
        self.elitism = elitism
        self.fitness_function = fitness_function
        self.job_queue = job_queue
        self.terminator = terminator
        self.evolution_state = EvolutionState([], [], [], False, 0)
        self.terminate_loop: bool = False
        self.events = events
        self.representation = []
        self.init_population = init_population
        self.population_size = population_size
        self.maximize = maximize

    def set_selection(self, selection: Selection) -> Self:
        self.selection = selection
        return self
    
    def set_crossover(self, crossover: Crossover) -> Self:
        self.crossover = crossover
        return self

    def set_mutation(self, mutation: Mutation) -> Self:
        self.mutation = mutation
        return self

    def use_preset(self, preset: OperatorsPreset) -> Self:
        self.selection = preset.selection
        self.crossover = preset.crossover
        self.mutation = preset.mutation
        return self

    def run(self) -> None:
        self.prepare_evolution_state()
        while not self.terminate_loop:
            self.loop()
            print(f"Snapshot: Generation {self.evolution_state.generation}, Population: {self.evolution_state.current_population}")
            # terminator.evaluate(self)
    
    def loop(self) -> None:
        self.step_selection()
        self.step_crossover()
        self.step_mutation()    

    def get_evolution_state(self) -> EvolutionState:
        return self.evolution_state
    
    def prepare_evolution_state(self) -> None:
        self.evolution_state.current_population = self.init_population
        self.evolution_state.maximize = self.maximize
        self.evolution_state.population_size = self.population_size

    def perform_crossover(self) -> None:
        self.evolution_state.current_population = self.crossover.cross(self.evolution_state.current_population)

    def perform_selection(self) -> None:
        self.evolution_state.current_population = self.selection.select(self.evolution_state.current_population)

    def perform_mutation(self) -> None:
        self.evolution_state.current_population = self.mutation.mutate(self.evolution_state.current_population)

    def step_selection(self) -> None:
        self.perform_selection()

    def step_crossover(self) -> None:
        self.perform_crossover()

    def step_mutation(self) -> None:
        self.perform_mutation()
