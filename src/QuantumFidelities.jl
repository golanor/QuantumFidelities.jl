module QuantumFidelities

using LinearAlgebra


# Single qubit Paulis
const I2 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

"""
    generate_pauli_basis(n_qubits::Int)

Generates the `d^2` (where `d=2^n_qubits`) operator basis of Pauli matrices
{P_j} such that `Tr(P_j P_k) = d * δ_jk`.
"""
function generate_pauli_basis(n_qubits::Int)
    n_qubits < 1 && throw(DomainError("Can't have less than 1 qubits."))
    single_qubit_basis = [I2, σx, σy, σz]
    
    basis = single_qubit_basis

    if n_qubits == 1
        return single_qubit_basis
    end
    
    for _ in 2:n_qubits
        new_basis = Matrix{ComplexF64}[]
        for op_i in single_qubit_basis
            for op_j in basis
                push!(new_basis, kron(op_i, op_j))
            end
        end
        basis = new_basis
    end
    
    basis
end




const TWO_QUBIT_PAULI_BASIS, TWO_QUBIT_PAULI_LABELS = let
    paulis = [I2, σx, σy, σz]
    labels = ["I", "X", "Y", "Z"]
    basis = 0.5 * generate_pauli_basis(2)
    label_strs = [labels[i] * "⊗" * labels[j] for i in 1:4 for j in 1:4]
    
    (basis, label_strs)
end


"""
    two_qubit_gate_fidelity(G1, G2)

Compute the two-qubit gate fidelity between two unitary operators `G1` and `G2`.
# Arguments
- `G1::AbstractMatrix{T}`: The first unitary operator
- `G2::AbstractMatrix{T}`: The second unitary operator
where T<:Number

# Returns
- `F`: The two-qubit gate fidelity

"""
function two_qubit_gate_fidelity(G1::AbstractMatrix{T}, G2::AbstractMatrix{T}) where T<:Number
    real((abs2(tr(G1' * G2)) + tr(G1' * G2 * G2' * G1)) / 20)
end


"""
    nielsen_average_gate_fidelity_open(
        U_ideal::AbstractMatrix{T},
        channel_func,
        pauli_basis::AbstractVector{<:AbstractMatrix{T}}
    ) where {T<:Number}

Calculate the average gate fidelity `F_avg` between an ideal unitary gate `U_ideal`
and a quantum channel `ℰ` (represented by `channel_func`).

This function implements the formula from Nielsen (2002), which is robust for
any completely positive (CP) map, including non-unital or non-trace-preserving
channels.

# Arguments
- `U_ideal`: The ideal unitary gate (e.g., a `d x d` matrix).
- `channel_func`: A callable function that represents the quantum channel `ℰ`.
  It must accept a single `d x d` matrix `P` and return the transformed
  matrix `ℰ(P)`.
- `pauli_basis`: A vector containing a complete, orthogonal basis of `d^2`
  Hermitian operators `{P_j}` for the space of `d x d` matrices. This basis
  **must** satisfy the orthogonality condition `Tr(P_j P_k) = d * δ_jk`. The
  standard two-qubit (d=4) Pauli basis (16 operators `σ_i ⊗ σ_j`) is a
  common example.

# Returns
- `F_avg`: The average gate fidelity, a real number `F_avg ∈ [0, 1]`.

# Formula
The fidelity is calculated using the formula:

`F_avg(ℰ, U) = [ Σ_j Tr( (U Pj U†)† ℰ(Pj) ) + d^2 ] / [ d^2(d + 1) ]`

where:
- `d` is the dimension of the Hilbert space (e.g., `d=4` for two qubits).
- `Pj` are the `d^2` elements of the `pauli_basis`.
- `ℰ(Pj)` is the result of applying the channel to `Pj`.
- `U` is the `U_ideal` gate.
- `†` denotes the conjugate transpose (adjoint).

# References
1.  M. A. Nielsen, "A simple formula for the average gate fidelity of a
    quantum-dynamical process," Phys. Lett. A, vol. 303, no. 4,
    pp. 249–252, Oct. 2002. (This is the source of the formula used).
2.  Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and
    Quantum Information.*
"""
function nielsen_average_gate_fidelity_open(
    U_ideal::AbstractMatrix{T},
    channel_func, # A callable (e.g., function)
    pauli_basis::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Number}

    d = size(U_ideal, 1)
    # Determine the underlying real type (e.g., Float64 or Float32)
    R = real(T)
    # Use R(d) to ensure type stability (e.g., 4.0, not 4)
    d_R = R(d)
    d_sq = R(d^2)

    fidelity_sum = zero(R) 
    U_ideal_dagger = U_ideal'

    for P_j in pauli_basis
        E_Pj = channel_func(P_j)
        U_Pj_Udag = U_ideal * P_j * U_ideal_dagger
        term = tr(U_Pj_Udag * E_Pj)
        fidelity_sum += real(term)
    end

    (fidelity_sum + d_sq) / (d_sq * (d_R + 1))
end

"""
    compute_process_matrix(
        input_states::AbstractVector{<:AbstractMatrix{T}},
        output_states::AbstractVector{<:AbstractMatrix{T}},
        pauli_basis::AbstractVector{<:AbstractMatrix{T}}
    ) where {T<:Number}

Compute the χ-matrix (process matrix) representation of a quantum channel
via Quantum Process Tomography (QPT).

This function solves the linear system of equations `A * χ_vec = b` that
defines the mapping from input to output states.

# Arguments
- `input_states`: A tomographically complete set of input density matrices
  (e.g., `Vector{Matrix{ComplexF64}}`).
- `output_states`: The corresponding output density matrices after the
  channel `ℰ` has been applied to each input state.
- `pauli_basis`: A complete, orthogonal operator basis (e.g., the Pauli
  operators) `{E_m}`. The basis must have `d^2` elements.

# Returns
- `χ`: The `d^2 x d^2` process matrix, which defines the channel via:
  `ℰ(ρ) = Σ_mn χ_mn E_m ρ E_n†`

# Notes
- The fidelity of this reconstruction depends heavily on the choice of
  `input_states`. They should form a tomographically complete basis.
- The returned `χ` matrix is enforced to be Hermitian, as is required
  for a physical process.

# References
1.  Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and
    Quantum Information.* (Chapter 8, Section 8.4)
2.  Poyatos, J. F., Cirac, J. I., & Zoller, P. (1997). "Complete
    characterization of a quantum process: The two-bit quantum gate."
    Phys. Rev. Lett., 78(2), 390–393.
"""
function compute_process_matrix(
    input_states::AbstractVector{<:AbstractMatrix{T}},
    output_states::AbstractVector{<:AbstractMatrix{T}},
    pauli_basis::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Number}

    n_basis = length(pauli_basis)
    n_states = length(input_states)
    @assert n_states > 0 "Must provide at least one input state."
    @assert n_states == length(output_states) "Must have same number of input and output states."
    
    d = size(input_states[1], 1)
    @assert n_basis == d^2 "Pauli basis must have d^2 elements."
    
    # Get the complex type (e.g., ComplexF64)
    CT = T
    
    # Build linear system A * χ_vec = b
    A = zeros(CT, d^2 * n_states, n_basis^2)
    b = zeros(CT, d^2 * n_states)
    
    for (s, (ρ_in, ρ_out)) in enumerate(zip(input_states, output_states))
        # Vectorize the output state ρ_out into the 'b' vector
        b_offset = (s - 1) * d^2
        b[b_offset+1:b_offset+d^2] = vec(ρ_out)

        # Build the 'A' matrix row by row
        for k in 1:(d^2) # k iterates over elements of ρ_out
            row = (s - 1) * d^2 + k
            
            # (i, j) are the matrix indices corresponding to k
            i = mod(k - 1, d) + 1
            j = div(k - 1, d) + 1
            
            for m in 1:n_basis, n in 1:n_basis
                col = (m - 1) * n_basis + n
                
                # E_m * ρ_in * E_n†
                temp = pauli_basis[m] * ρ_in * pauli_basis[n]'
                
                # The (i,j) element of this operation
                A[row, col] = temp[i, j]
            end
        end
    end
    
    # Solve the linear system
    χ_vec = A \ b
    χ = reshape(χ_vec, n_basis, n_basis)
    
    # Enforce Hermiticity, as χ must be Hermitian
    (χ + χ') / 2
end


"""
    nielsen_from_process_matrix(
        U_ideal::AbstractMatrix{T},
        χ_process::AbstractMatrix{T},
        pauli_basis::AbstractVector{<:AbstractMatrix{T}}
    ) where {T<:Number}

Calculate average gate fidelity using the χ-matrix (process matrix).

This function serves as a wrapper. It reconstructs the channel function `ℰ`
from the `χ_process` matrix and then calls
`nielsen_average_gate_fidelity_open`.

# Arguments
- `U_ideal`: The ideal unitary gate (e.g., a `d x d` matrix).
- `χ_process`: The `d^2 x d^2` process matrix.
- `pauli_basis`: The operator basis `{E_m}` corresponding to `χ_process`.

# Returns
- `F_avg`: The average gate fidelity, `F_avg ∈ [0, 1]`.
"""
function nielsen_average_gate_fidelity_open(
    U_ideal::AbstractMatrix{T},
    χ_process::AbstractMatrix{T},
    pauli_basis::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Number}

    d = size(U_ideal, 1)
    n_basis = length(pauli_basis)
    @assert n_basis == d^2 "Pauli basis must have d^2 elements."
    @assert size(χ_process) == (n_basis, n_basis) "χ_process matrix has incorrect dimensions."

    # Get the correct typing.
    CT = T

    # Define the channel ℰ(ρ) from the χ-matrix
    # ℰ(ρ) = Σ_mn χ_mn E_m ρ E_n†
    function channel_from_chi(P::AbstractMatrix)
        E_P = zeros(CT, d, d)
        for m in 1:n_basis, n in 1:n_basis
            E_P .+= χ_process[m, n] .* (pauli_basis[m] * P * pauli_basis[n]')
        end
        E_P
    end
    
    nielsen_average_gate_fidelity_open(U_ideal, channel_from_chi, pauli_basis)
end


export nielsen_average_gate_fidelity_open, two_qubit_gate_fidelity, compute_process_matrix, generate_pauli_basis
end
