using Test
using LinearAlgebra

# Robustly include the module from the parent's src/ directory
# This works whether you run `julia tests/tests.jl` from the root
# or `cd tests` and run `julia tests.jl`.
include(joinpath(@__DIR__, "..", "src", "QuantumFidelities.jl"))

# Bring the module and its exported functions into scope
using .QuantumFidelities

# ------------------------------------------------------------------
#  HELPER FUNCTIONS FOR TESTING
# ------------------------------------------------------------------


"""
    _generate_qpt_states(n_qubits::Int)

Generates a tomographically complete set of 4^n_qubits input states
for single-qubit QPT.
States are |0>, |1>, |+>, |+i>
"""
function _generate_qpt_states(n_qubits::Int)
    # Define single-qubit states
    ket0 = ComplexF64[1, 0]
    ket1 = ComplexF64[0, 1]
    ketP = (ket0 + ket1) / sqrt(2)
    ketY = (ket0 + im * ket1) / sqrt(2)
    
    single_qubit_states = [
        ket0 * ket0', # |0><0|
        ket1 * ket1', # |1><1|
        ketP * ketP', # |+><+|
        ketY * ketY'  # |+i><+i|
    ]
    
    # Initialize basis with a placeholder
    states = [Matrix{ComplexF64}(undef, 0, 0)]
    
    for i in 1:n_qubits
        if i == 1
            states = single_qubit_states
        else
            new_states = []
            for state_i in single_qubit_states
                for state_j in states
                    push!(new_states, kron(state_i, state_j))
                end
            end
            states = new_states
        end
    end
    
    states
end

# ------------------------------------------------------------------
#  TEST SUITE
# ------------------------------------------------------------------
@testset "Nielsen Average Gate Fidelity Tests" begin

    @testset "Test 1: Perfect Fidelity (Ideal Channel)" begin
        d = 4 # Two qubits
        basis = generate_pauli_basis(2)
        @test length(basis) == d^2 # 16
        
        # Use CNOT as the ideal gate
        CNOT = Matrix{ComplexF64}([
            1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0
        ])
        
        # Define the ideal channel: ℰ(P) = U P U†
        ideal_channel(P) = CNOT * P * CNOT'
        
        # Test the function from the imported module
        F_avg = nielsen_average_gate_fidelity_open(CNOT, ideal_channel, basis)
        
        # The fidelity should be exactly 1.0
        @test F_avg ≈ 1.0
    end
    
    @testset "Test 2: Depolarizing Channel (Two Qubits)" begin
        d = 4
        basis = generate_pauli_basis(2)
        
        # Test against the Identity gate
        U_ideal = Matrix{ComplexF64}(I, d, d)
        
        p = 0.9 # Fidelity parameter
        
        # Depolarizing channel: ℰ(P) = p*P for traceless P, ℰ(I) = I
        # This works because our basis is {I, P_j} where Tr(P_j)=0 for j>0
        function depolarizing_channel(P)
            if P ≈ I
                return P # ℰ(I) = I
            else
                # For all other basis elements, Tr(P)=0
                # ℰ(P_j) = p * P_j
                return p * P
            end
        end
        
        F_avg_calc = nielsen_average_gate_fidelity_open(U_ideal, depolarizing_channel, basis)
        
        # Analytical formula for avg gate fidelity of depolarizing channel
        # F_g (process fidelity) = p + (1-p)/d^2
        # F_avg = (d * F_g + 1) / (d+1)
        F_g_analytic = p + (1 - p) / d^2
        F_avg_analytic = (d * F_g_analytic + 1) / (d + 1)
        
        # For p=0.9, d=4:
        # F_g = 0.9 + 0.1/16 = 0.90625
        # F_avg = (4 * 0.90625 + 1) / 5 = (3.625 + 1) / 5 = 4.625 / 5 = 0.925
        
        @test F_avg_analytic ≈ 0.925
        @test F_avg_calc ≈ F_avg_analytic
    end
    
    @testset "Test 3: Phase Damping Channel (Single Qubit)" begin
        d = 2 # Single qubit
        basis = generate_pauli_basis(1)
        @test length(basis) == d^2 # 4
        
        # Test against Identity
        U_ideal = Matrix{ComplexF64}(I, d, d)
        
        γ = 0.1 # Damping parameter
        s = sqrt(1 - γ)
        
        K0 = Matrix{ComplexF64}([1 0; 0 s])
        K1 = Matrix{ComplexF64}([0 0; 0 sqrt(γ)])
        
        phase_damping_channel(P) = K0 * P * K0' + K1 * P * K1'
        
        F_avg_calc = nielsen_average_gate_fidelity_open(U_ideal, phase_damping_channel, basis)
        
        # Analytical calculation (from derivation):
        # ℰ(I) = I
        # ℰ(X) = s*X
        # ℰ(Y) = s*Y
        # ℰ(Z) = Z
        # fidelity_sum = Tr(I*ℰ(I)) + Tr(X*s*X) + Tr(Y*s*Y) + Tr(Z*ℰ(Z))
        #              = Tr(I^2) + Tr(X*s*X) + Tr(Y*s*Y) + Tr(Z^2)
        #              = d + s*d + s*d + d = 2d + 2d*s
        # F_avg = (fidelity_sum + d^2) / (d^2 * (d+1))
        # d=2, d^2=4:
        # fidelity_sum = 4 + 4s
        # F_avg = (4 + 4s + 4) / (4 * 3) = (8 + 4s) / 12 = (2 + s) / 3
        
        F_avg_analytic = (2 + s) / 3
        
        @test F_avg_calc ≈ F_avg_analytic
    end
end

@testset "Two Qubit Gate Fidelity Tests" begin
    
    @testset "Test 1: Identical Gates" begin
        # Test that the same gate returns fidelity 1.0
        CNOT = Matrix{ComplexF64}([
            1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0
        ])
        
        @test two_qubit_gate_fidelity(CNOT, CNOT) ≈ 1.0
    end
    
    @testset "Test 2: CNOT vs CNOT * (Z ⊗ I)" begin
        # CNOT gate
        CNOT = Matrix{ComplexF64}([
            1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0
        ])
        
        # Z ⊗ I gate
        Z_tensor_I = Matrix{ComplexF64}([
            1 0 0 0
            0 1 0 0
            0 0 -1 0
            0 0 0 -1
        ])
        
        G2 = CNOT * Z_tensor_I
        
        # Analytically computed fidelity
        # F = (|Tr(G1† G2)|^2 + Tr(G1† G2 G2† G1)) / 20
        # For CNOT and CNOT*(Z⊗I):
        # Tr(CNOT† * CNOT * Z⊗I) = Tr(Z⊗I) = 0
        # Tr(CNOT† * CNOT * Z⊗I * Z⊗I† * CNOT† * CNOT) = Tr(I_4) = 4
        # F = (0 + 4) / 20 = 0.2
        expected_fidelity = 0.2
        
        @test two_qubit_gate_fidelity(CNOT, G2) ≈ expected_fidelity
    end
    
    @testset "Test 3: Identity vs Hadamard ⊗ Hadamard" begin
        # Identity gate
        I4 = Matrix{ComplexF64}(I, 4, 4)
        
        # Hadamard ⊗ Hadamard gate
        H = (1/sqrt(2)) * Matrix{ComplexF64}([1 1; 1 -1])
        H_tensor_H = kron(H, H)
        
        # Analytically computed fidelity
        # F = (|Tr(I4† * H⊗H)|^2 + Tr(I4† * H⊗H * (H⊗H)† * I4)) / 20
        # Tr(H⊗H) = Tr(H)^2 = 0
        # Tr((H⊗H) * (H⊗H)†) = Tr(I_4) = 4
        # F = (0 + 4) / 20 = 0.2
        expected_fidelity = 0.2
        
        @test two_qubit_gate_fidelity(I4, H_tensor_H) ≈ expected_fidelity
    end
end

@testset "Quantum Process Tomography & Chi Fidelity" begin

    # --- Single Qubit (d=2) Setup ---
    d = 2
    basis = generate_pauli_basis(1) # I, X, Y, Z
    qpt_states = _generate_qpt_states(1) # |0><0|, |1><1|, |+><+|, |+i><+i|
    
    I_d2 = Matrix{ComplexF64}(I, d, d)
    X_d2 = Matrix{ComplexF64}([0 1; 1 0])

    @testset "Test 4: QPT - Identity Channel" begin
        # The ideal channel is just identity
        ideal_channel(P) = I_d2 * P * I_d2'
        
        input_states = qpt_states
        output_states = [ideal_channel(P) for P in input_states]
        
        χ = compute_process_matrix(input_states, output_states, basis)
        
        # For an ideal Identity channel ℰ(ρ) = I ρ I,
        # and basis[1] = I, we expect χ[1, 1] = 1
        # and all other elements to be 0.
        χ_ideal = zeros(4, 4)
        χ_ideal[1, 1] = 1.0
        
        @test χ ≈ χ_ideal atol=1e-14
    end

    @testset "Test 5: QPT - Bit-Flip Channel" begin
        # The ideal channel is a bit-flip
        bit_flip_channel(P) = X_d2 * P * X_d2'
        
        input_states = qpt_states
        output_states = [bit_flip_channel(P) for P in input_states]
        
        χ = compute_process_matrix(input_states, output_states, basis)
        
        # For an ideal Bit-Flip channel ℰ(ρ) = X ρ X,
        # and basis[2] = X, we expect χ[2, 2] = 1
        # and all other elements to be 0.
        χ_ideal = zeros(ComplexF64, 4, 4)
        χ_ideal[2, 2] = 1.0
        
        @test χ ≈ χ_ideal atol=1e-14
    end
    
    @testset "Test 6: End-to-End Fidelity from Chi (Ideal)" begin
        # 1. Get the χ matrix for the Identity channel
        ideal_channel(P) = I_d2 * P * I_d2'
        input_states = qpt_states
        output_states = [ideal_channel(P) for P in input_states]
        χ_identity = compute_process_matrix(input_states, output_states, basis)
        
        # 2. Calculate fidelity using the χ matrix
        # The ideal gate is Identity
        F_avg = nielsen_average_gate_fidelity_open(I_d2, χ_identity, basis)
        
        # The fidelity should be 1.0
        @test F_avg ≈ 1.0
    end
    
    @testset "Test 7: End-to-End Fidelity from Chi (Bit-Flip)" begin
        # 1. Get the χ matrix for the Bit-Flip channel
        bit_flip_channel(P) = X_d2 * P * X_d2'
        input_states = qpt_states
        output_states = [bit_flip_channel(P) for P in input_states]
        χ_bitflip = compute_process_matrix(input_states, output_states, basis)
        
        # 2. Calculate fidelity *against the bit-flip gate*
        # The ideal gate is X
        F_avg = nielsen_average_gate_fidelity_open(X_d2, χ_bitflip, basis)
        
        # The fidelity should be 1.0 (the channel perfectly matches the gate)
        @test F_avg ≈ 1.0
        
        # 3. Calculate fidelity *against the identity gate*
        # The ideal gate is I
        F_avg_vs_I = nielsen_average_gate_fidelity_open(I_d2, χ_bitflip, basis)
        
        # Analytic result for F_avg(X, I) is 1/3
        @test F_avg_vs_I ≈ 1/3
    end
end
