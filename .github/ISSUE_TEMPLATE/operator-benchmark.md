---
name: Operator benchmark
about: Benchmark existing or proposed genetic operators.
title: ''
labels: benchmark
assignees: ''

---

## Benchmark Goal

What is the purpose of this benchmark?

- [ ] Compare different operators  
- [ ] Evaluate a newly proposed operator  
- [ ] Stress-test under large population sizes  
- [ ] Test edge cases (e.g. degenerate inputs)  
- [ ] Other (please specify): `_________`

## Problem Instance(s)

Which optimization problems are used in this benchmark?

- [ ] OneMax (binary)  
- [ ] Rastrigin (real-valued)  
- [ ] Knapsack Problem  
- [ ] Traveling Salesman Problem (TSP)  
- [ ] Custom (describe below)  

**Problem details (if custom):**
```text
Name: MyCustomProblem  
Dimensionality: 50  
Constraints: ...
```

## Operators Under Test

List the operators involved in this benchmark:

## Evaluation Criteria

What metrics will be used to compare operators?
- [ ] Final solution quality (fitness)
- [ ] Convergence speed (generations/time)
- [ ] Runtime performance
- [ ] Reproducibility / stability (e.g. stddev over runs)
- [ ] Other: `_________`

## Benchmark Configuration

Basic setup for the test:
- [ ] Population size: `____`
- [ ] Number of generations: `____`
- [ ] Selection method: `____`
- [ ] Representation type: binary / float / permutation
- [ ] Random seed(s): `____`

## Runtime Environment
- Python version: X.X.X
- OS: Windows / Linux / macOS
- CPU/GPU: Intel i7 / AMD Ryzen / Apple M1 / ...
- Library version: 0.X.X

## Expected Output

What will the benchmark generate?
- [ ] Tabular results
- [ ] Plots (fitness over time, variance, etc.)
- [ ] Logs or CSV output
- [ ] JSON reports
