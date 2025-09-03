# Emergent Geometry from Quantum Correlations: A Minimal Solvable Model

This repository contains the official Python simulation code for the theoretical physics paper: **"Fundamentals of Emergent Geometry: A Minimal Solvable Model and its Phenomenology"**.

## Overview

The script `simulation_code.py` provides a numerical implementation of the 3-qubit toy model described in the paper. The model investigates the hypothesis that spacetime geometry is an emergent property derived from the entanglement structure of an underlying quantum system.

The simulation performs the following key steps:
1.  Constructs the interaction Hamiltonian for a 3-qubit system with a variable asymmetry parameter, `eta`.
2.  Numerically diagonalizes the Hamiltonian to find the exact ground state for each `eta`.
3.  Calculates the bipartite quantum mutual information between all pairs of qubits from the ground state.
4.  Computes the emergent distances based on the "information distance" postulate.
5.  Determines the scalar curvature of the emergent discrete geometry.

The main output is a plot of the emergent curvature as a function of the entanglement asymmetry, revealing a rich phase structure including stiff-geometric, resonant, and non-geometric regimes.

## Requirements

The code is written in Python 3 and requires the following standard scientific libraries:
- `numpy`
- `scipy`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy scipy matplotlib
