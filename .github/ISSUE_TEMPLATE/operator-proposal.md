---
name: Operator proposal
about: Propose a new genetic operator (e.g. crossover, mutation, selection)
title: ''
labels: feature
assignees: ''

---

## Operator Type

What type of operator is being proposed?

- [ ] Selection
- [ ] Crossover
- [ ] Mutation
- [ ] Other (please specify): `_________`

## Operator Name

What is the commonly used name for this operator (if any)?

> Example: "Uniform Crossover", "Tournament Selection", "Gaussian Mutation"

## Source / Background

Where does this operator come from?

- [ ] Academic paper / textbook
- [ ] Existing implementation
- [ ] Original idea / experimental

Please provide references or links if available:

## Motivation

Why should this operator be added?  
How does it improve upon existing options?

## Pseudocode or Logic

Please describe the algorithm briefly or provide pseudocode:
```text
Input: Parent A, Parent B
For each gene:
    If random() < 0.5:
        Child[i] = A[i]
    else:
        Child[i] = B[i]
