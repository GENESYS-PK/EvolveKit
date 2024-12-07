from typing import List
from core.expression import Job
from core.evolution import Evolution

class JobManager:
    def __init__(self, jobs: List[Job] = None):
        self.jobs = jobs if jobs is not None else []

    def add_job(self, job: Job) -> None:
        self.jobs.append(job)

    def remove_job(self, job: Job) -> None:
        self.jobs.remove(job)

    def evaluate_jobs(self, evolution: Evolution) -> None:
        for job in self.jobs:
            if job.evaluate(evolution):
                job.handle(evolution)