# QuantumFidelities.jl

[![CI](https://github.com/orgolan/QuantumFidelities.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/orgolan/QuantumFidelities.jl/actions/workflows/CI.yml)

A Julia package for calculating various quantum fidelity measures, with a focus on quantum process tomography and gate fidelity estimation.

## Overview

`QuantumFidelities.jl` provides a set of tools for characterizing and benchmarking quantum gates and processes. It allows you to:

*   Compute the fidelity of quantum gates.
*   Characterize quantum channels using process tomography (χ-matrix).
*   Represent quantum processes with Choi matrices.
*   Calculate average gate fidelity for both ideal and noisy quantum channels.

This package is useful for researchers and engineers working on quantum computing, especially in the areas of quantum device characterization and benchmarking.

## Features

*   **Gate Fidelity:** Calculate the fidelity between two unitary gates.
*   **Average Gate Fidelity:** Implements Nielsen's formula for average gate fidelity of a quantum process.
*   **Quantum Process Tomography:** Compute the process matrix (χ-matrix) from a set of input and output states.
*   **Choi Matrix:**
    *   Compute the Choi matrix for an ideal unitary gate.
    *   Convert a process matrix (χ-matrix) to a Choi matrix.
*   **Fidelity from Choi Matrix:**
    *   Calculate average gate fidelity from ideal and actual Choi matrices.
    *   Compute the state fidelity between two Choi matrices.
*   **Pauli Basis Generation:** Generate a basis of Pauli operators for a given number of qubits.
*   **Channel Construction:** Create a quantum channel function from input-output state mappings.

## Installation

To install `QuantumFidelities.jl`, open the Julia REPL and run:

```julia
using Pkg
Pkg.add("QuantumFidelities")
```

## Usage

Here is a simple example of how to calculate the average gate fidelity of a noisy quantum channel.

First, let's define an ideal CNOT gate and a noisy channel that applies the CNOT gate with some depolarizing noise.

```julia
using QuantumFidelities
using LinearAlgebra

# Ideal CNOT gate
CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

# Define a noisy channel (e.g., depolarizing channel)
function noisy_cnot_channel(ρ, p=0.9)
    # Apply ideal CNOT
    perfect_output = CNOT * ρ * CNOT'
    
    # Apply depolarizing noise
    d = size(ρ, 1)
    noisy_output = p * perfect_output + (1 - p) * I(d) / d
    
    return noisy_output
end

# Generate the 2-qubit Pauli basis
pauli_basis = generate_pauli_basis(2)

# Calculate the average gate fidelity
fidelity = nielsen_average_gate_fidelity_open(CNOT, noisy_cnot_channel, pauli_basis)

println("Average Gate Fidelity: ", fidelity)
```

## API Reference

The following functions are exported by `QuantumFidelities.jl`:

*   `generate_pauli_basis(n_qubits::Int)`: Generates the Pauli operator basis for `n_qubits`.
*   `two_qubit_gate_fidelity(G1, G2)`: Computes the gate fidelity between two two-qubit gates `G1` and `G2`.
*   `nielsen_average_gate_fidelity_open(U_ideal, channel_func, pauli_basis)`: Calculates the average gate fidelity of a quantum channel `channel_func` with respect to an ideal unitary `U_ideal`.
*   `compute_process_matrix(input_states, output_states, pauli_basis)`: Computes the χ-matrix for a quantum process from input-output state pairs.
*   `compute_choi_matrix_ideal(U)`: Computes the Choi matrix for an ideal unitary gate `U`.
*   `compute_choi_from_chi(χ_process, pauli_basis)`: Converts a process matrix `χ_process` to a Choi matrix.
*   `choi_average_gate_fidelity(choi_ideal, choi_actual)`: Calculates the average gate fidelity from two Choi matrices.
*   `choi_state_fidelity(choi_ideal, choi_actual)`: Computes the state fidelity between two Choi matrices.
*   `create_channel_from_mappings(input_states, output_states)`: Creates a channel function from a set of input-output state mappings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.