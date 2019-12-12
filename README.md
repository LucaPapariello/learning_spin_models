# Machine learning spin models

### Introduction

This repository contains code to apply some machine learning (ML) techniques to two two-dimensional lattice models: 

* the Ising model
* the Ising lattice gauge theory.

### Structure of the repository

The two models are divided into the two homonymous folders, i.e. `/Ising`, and `/lattice_gauge_theory`. In both of them, the Python script `create_configurations.py` generates (Monte Carlo) samples for the corresponding model and stores them in the `/configs` folder. These spin configurations are the synthetic data used to train the ML models &mdash; this is done in the corresponding Jupyter notebooks.

### Disclaimers

The code here contained is meant to show some cool things that can be done when applying machine learning techniques to physics. However, this code is not used for any publication and there is no guarantee of correctness or convergence of results. 

### References

The study of these models in relation to ML is motivated (i) by the paper of [J. Carrasquilla and R. Melko, Nature Phys. 13, 431–434 (2017)](https://www.nature.com/articles/nphys4035), and (ii) by the journal club held at ETH Zurich in 2018, organized by Sebastian Huber, Mark Fischer, and Maciej Koch-Janusz. The content of this repository builds on Mark's code and presentation.
