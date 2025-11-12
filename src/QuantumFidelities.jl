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


# ------------------------------------------------------------------
#  NEW FUNCTIONS (Choi Matrix and Fidelities)
# ------------------------------------------------------------------

"""
    compute_choi_matrix_ideal(U::AbstractMatrix{T}) where {T<:Complex}

Compute the Choi matrix `J` for an ideal unitary gate `U`.

The Choi matrix ``J`` for a channel ``\\mathcal{E}`` is defined via the
Jamiolkowski isomorphism. For a unitary channel ``\\mathcal{E}_U(\\rho) = U \\rho U^\\dagger``,
the Choi matrix can be computed directly from the unitary `U`.

This function uses the column-stacking (`vec`) convention, where:
``J(\\mathcal{E}_U) = |U\\rangle\\rangle \\langle\\langle U|``
where ``|U\\rangle\\rangle = \\text{vec}(U)`` is the ``d^2 \\times 1`` vectorization of `U`.

The resulting matrix ``J`` is ``d^2 \\times d^2`` and has ``\\text{Tr}(J) = d``.
It is Hermitian and positive semi-definite, with rank 1.

# Arguments
- `U::AbstractMatrix{T}`: The ideal unitary operator (``d \\times d``).

# Returns
- `J::Matrix{T}`: The corresponding Choi matrix (``d^2 \\times d^2``).
"""
function compute_choi_matrix_ideal(U::AbstractMatrix{T}) where {T<:Complex}
    # vec(U) performs column-stacking, creating a d^2 x 1 vector
    U_vec = vec(U)
    
    # J = |U>> <<U|
    U_vec * U_vec'
end

"""
    compute_choi_from_chi(
        χ_process::AbstractMatrix{T},
        pauli_basis::AbstractVector{<:AbstractMatrix{T}}
    ) where {T<:Complex}

Compute the Choi matrix `J` from a process matrix `χ`.

The Choi matrix ``J`` and process matrix ``\\chi`` are related by a
change of basis. This function implements the transformation:
``J = \\sum_{m,n} \\chi_{mn} \\text{vec}(E_m) \\text{vec}(E_n)^\\dagger``
where ``|E_m\\rangle\\rangle = \\text{vec}(E_m)`` is the vectorized form of the
``m``-th basis operator.

This function is designed to be used with `compute_process_matrix`.

# Arguments
- `χ_process::AbstractMatrix{T}`: The ``d^2 \\times d^2`` process matrix.
- `pauli_basis::AbstractVector{<:AbstractMatrix{T}}`: The ``d^2`` operator
  basis elements `{E_m}` corresponding to `χ_process`.

# Returns
- `J::Matrix{T}`: The corresponding Choi matrix (``d^2 \\times d^2``).
"""
function compute_choi_from_chi(
    χ_process::AbstractMatrix{T},
    pauli_basis::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Complex}
    d_sq = size(χ_process, 1)
    d = isqrt(d_sq)
    @assert d^2 == d_sq "χ_process matrix must be d^2 x d^2."
    @assert length(pauli_basis) == d_sq "Basis must have d^2 elements."
    
    CT = T
    choi = zeros(CT, d_sq, d_sq)
    
    # Pre-calculate vec(E_m) for all m
    vec_basis = [vec(P) for P in pauli_basis]
    
    for m in 1:d_sq, n in 1:d_sq
        # J += χ_mn * |E_m>> <<E_n|
        # |E_m>> = vec_basis[m]
        # <<E_n| = vec_basis[n]'
        choi .+= χ_process[m, n] .* (vec_basis[m] * vec_basis[n]')
    end
    
    choi # implicit return
end

"""
    choi_average_gate_fidelity(
        choi_ideal::AbstractMatrix{T},
        choi_actual::AbstractMatrix{T}
    ) where {T<:Complex}

Calculate the average gate fidelity ``F_{avg}`` from two Choi matrices.

This function first computes the process fidelity ``F_{process}`` and then
converts it to the average gate fidelity ``F_{avg}``.

The process fidelity is defined as:
``F_{process}(\\mathcal{E}, \\mathcal{U}) = \\frac{1}{d^2} \\text{Tr}(J_U^\\dagger J_\\mathcal{E})``
where ``J_U`` and ``J_\\mathcal{E}`` are the Choi matrices, normalized to have trace ``d``.

The average gate fidelity is then:
``F_{avg} = \\frac{d \\cdot F_{process} + 1}{d+1}``

# Arguments
- `choi_ideal::AbstractMatrix{T}`: The ``d^2 \\times d^2`` Choi matrix for the
  ideal unitary gate. Assumed to have ``\\text{Tr}(J) \\approx d``.
- `choi_actual::AbstractMatrix{T}`: The ``d^2 \\times d^2`` Choi matrix for the
  actual quantum process. Its trace is ``\\text{Tr}(\\mathcal{E}(\\mathbb{I}))``.

# Returns
- `F_avg::R`: The average gate fidelity, a real number ``R \\in [0, 1]``.

# References
1.  Horodecki, M., Horodecki, P., & Horodecki, R. (1999). General
    teleportation channel, singlet fraction, and quasidistillation.
    Phys. Rev. A, 60(3), 1888–1898.
2.  Nielsen, M. A. (2002). A simple formula for the average gate
    fidelity of a quantum-dynamical process. Phys. Lett. A, 303(4),
    249–252.
"""
function choi_average_gate_fidelity(
    choi_ideal::AbstractMatrix{T},
    choi_actual::AbstractMatrix{T}
) where {T<:Complex}
    d_sq = size(choi_ideal, 1)
    d = isqrt(d_sq)
    @assert d^2 == d_sq "Choi matrices must be d^2 x d^2."
    @assert size(choi_actual) == (d_sq, d_sq) "Choi matrices must have same dimensions."

    R = real(T) # Get the underlying real type
    d_R = R(d)

    # Traces should be real for Hermitian matrices
    tr_ideal = real(tr(choi_ideal))
    tr_actual = real(tr(choi_actual))
    
    # Avoid division by zero
    if abs(tr_ideal) < eps(R) || abs(tr_actual) < eps(R)
        return zero(R)
    end

    # Normalize Choi matrices such that Tr(J) = d
    # For ideal J_U, tr_ideal = d.
    # For actual J_E, tr_actual = Tr(E(I)).
    choi_ideal_normalized = choi_ideal .* (d_R / tr_ideal)
    choi_actual_normalized = choi_actual .* (d_R / tr_actual)
    
    # Process fidelity
    F_process = real(tr(choi_ideal_normalized' * choi_actual_normalized)) / (d_R^2)
    
    # Convert to average gate fidelity
    F_avg = (d_R * F_process + 1) / (d_R + 1)
    
    clamp(F_avg, zero(R), one(R))
end

"""
    choi_state_fidelity(
        choi_ideal::AbstractMatrix{T},
        choi_actual::AbstractMatrix{T}
    ) where {T<:Complex}

Calculate the state fidelity ``F(\\rho, \\sigma)`` between two Choi matrices.

This function treats the Choi matrices ``J`` as states (density matrices)
by normalizing them: ``\\rho_J = J / \\text{Tr}(J)``. It then computes
Uhlmann's state fidelity between them.

``F(\\rho, \\sigma) = \\left( \\text{Tr} \\sqrt{\\sqrt{\\rho} \\sigma \\sqrt{\\rho}} \\right)^2``

If the ideal state ``\\rho_{ideal}`` is pure (``\\text{Tr}(\\rho^2) = 1``), this
simplifies to ``F(\\rho_{ideal}, \\sigma) = \\text{Tr}(\\rho_{ideal} \\sigma)``.

# Arguments
- `choi_ideal::AbstractMatrix{T}`: The ``d^2 \\times d^2`` "ideal" Choi matrix.
- `choi_actual::AbstractMatrix{T}`: The ``d^2 \\times d^2`` "actual" Choi matrix.

# Returns
- `F_state::R`: The Uhlmann state fidelity, a real number ``R \\in [0, 1]``.
"""
function choi_state_fidelity(
    choi_ideal::AbstractMatrix{T},
    choi_actual::AbstractMatrix{T}
) where {T<:Complex}
    d_sq = size(choi_ideal, 1)
    @assert size(choi_actual) == (d_sq, d_sq) "Choi matrices must have same dimensions."
    
    R = real(T)
    
    tr_ideal = real(tr(choi_ideal)) # Trace should be real
    tr_actual = real(tr(choi_actual))

    if tr_ideal < eps(R) || tr_actual < eps(R)
        # If either trace is ~zero, fidelity is zero
        return zero(R)
    end
    
    # Normalize to density matrices (trace 1)
    ρ_ideal = choi_ideal ./ tr_ideal
    ρ_actual = choi_actual ./ tr_actual
    
    # Check for purity of the ideal state
    # The ideal Choi state J_U/d is pure, so this branch will be taken.
    # Use a tolerance scaled by dimension
    is_pure = abs(tr(ρ_ideal * ρ_ideal) - one(R)) < eps(R) * d_sq

    if is_pure
        # Pure state case: F(ρ_pure, σ) = Tr(ρ_pure * σ)
        # We take real() because fidelity is real and Tr might have small imag noise
        real(tr(ρ_ideal * ρ_actual))
    else
        # Mixed state case: Uhlmann's fidelity
        # F(ρ, σ) = (Tr[sqrt(sqrt(ρ) * σ * sqrt(ρ))])^2
        
        # We wrap in Hermitian() to tell sqrt() to use a stable algorithm
        # for Hermitian matrices, ensuring real eigenvalues.
        sqrt_ideal = sqrt(Hermitian(ρ_ideal))
        temp = sqrt_ideal * ρ_actual * sqrt_ideal
        
        # F_uhllmann = tr(sqrt(Hermitian(temp)))
        # `tr(sqrt(Hermitian(matrix)))` sums the square roots of the eigenvalues
        F_uhllmann = tr(sqrt(Hermitian(temp)))
        
        # Fidelity is the square of this trace.
        # Take real() to drop any floating point imaginary noise
        real(F_uhllmann^2)
    end
end

"""
    create_channel_from_mappings(
        input_states::AbstractVector{<:AbstractMatrix{T}},
        output_states::AbstractVector{<:AbstractMatrix{T}}
    ) where {T<:Complex}

Creates a quantum channel function `E(ρ)` from a set of known input-output
state mappings.

This function computes the ``d^2 \\times d^2`` superoperator ``\\mathcal{L}`` that
represents the channel via ``\\text{vec}(\\mathcal{E}(\\rho)) = \\mathcal{L} \\cdot \\text{vec}(\\rho)``.

It solves the linear system ``B = \\mathcal{L} A`` in the least-squares sense, where:
- `A = [vec(input_states[1]), ..., vec(input_states[N])]`
- `B = [vec(output_states[1]), ..., vec(output_states[N])]`

The solution is ``\\mathcal{L} = B \\cdot \\text{pinv}(A)``.

The returned function `channel_func(ρ)` is a fast closure that applies this
pre-computed superoperator ``\\mathcal{L}``. This function is suitable for passing to
`nielsen_average_gate_fidelity_open`.

# Arguments
- `input_states`: A tomographically complete set of input density matrices.
- `output_states`: The corresponding output density matrices.

# Returns
- `channel_func`: A function `E(ρ)` that takes a ``d \\times d`` density matrix
  and returns the ``d \\times d`` output matrix ``\\mathcal{E}(\\rho)``.
"""
function create_channel_from_mappings(
    input_states::AbstractVector{<:AbstractMatrix{T}},
    output_states::AbstractVector{<:AbstractMatrix{T}}
) where {T<:Complex}
    n_states = length(input_states)
    @assert n_states > 0 "Must provide at least one input state."
    @assert n_states == length(output_states) "Must have same number of input and output states."
    
    d = size(input_states[1], 1)
    d_sq = d^2
    
    # Construct A and B matrices
    CT = T
    A = zeros(CT, d_sq, n_states)
    B = zeros(CT, d_sq, n_states)
    
    for i in 1:n_states
        A[:, i] = vec(input_states[i])
        B[:, i] = vec(output_states[i])
    end
    
    # Solve for the superoperator L
    # L = B * A_pinv
    L_superop = B * pinv(A)
    
    # Return a fast, pre-computed channel function
    function channel_func(ρ::AbstractMatrix)
        @assert size(ρ) == (d, d) "Input matrix has incorrect dimensions."
        ρ_vec = vec(ρ)
        E_ρ_vec = L_superop * ρ_vec
        reshape(E_ρ_vec, d, d)
    end
    
    channel_func # Return the closure
end


export nielsen_average_gate_fidelity_open, two_qubit_gate_fidelity, compute_process_matrix, generate_pauli_basis
export compute_choi_matrix_ideal, compute_choi_from_chi, choi_average_gate_fidelity, choi_state_fidelity
export create_channel_from_mappings

end
