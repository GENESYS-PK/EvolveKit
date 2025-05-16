import datetime
from core.evolution.Evolution import Evolution
from core.expression.Job import Job
from typing import Callable

class TimeJob(Job):
    def __init__(self, action: Callable[[Evolution], None], interval: datetime.timedelta):
        super().__init__(action)
        self.creation_date = datetime.datetime.now()
        self.interval = interval

    def evaluate(self, evolution: Evolution) -> bool:
        return datetime.datetime.now() - self.creation_date >= self.interval