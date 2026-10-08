"""
vqe_mitiq_zne.py
Day 4 Script: Apply Zero-Noise Extrapolation (ZNE) to mitigate hardware noise 
on the H2 molecular energy calculation using native Gate Folding and Extrapolation.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE

# Qiskit Aer Primitives and Noise Imports
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

def build_custom_noise_model():
    """Constructs realistic NISQ noise model."""
    noise_model = NoiseModel()
    
    # 1. Single-qubit depolarizing error (0.5%)
    p_1q = 0.005
    err_1q = depolarizing_error(p_1q, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ["ry", "rz", "x", "sx"])
    
    # 2. Two-qubit CZ depolarizing error (2.0%)
    p_2q = 0.02
    err_2q = depolarizing_error(p_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cz"])
    
    # 3. Readout error (2.0%)
    p_read = 0.02
    readout_err = ReadoutError([[1 - p_read, p_read], [p_read, 1 - p_read]])
    noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model

def fold_circuit_global(circuit, scale_factor=1.0):
    """
    Scales noise by folding gates (g -> g g_dagger g).
    For scale_factor = 3.0, every foldable gate is replaced by g g_dagger g (3x depth/noise).
    For scale_factor = 2.0, half the foldable gates are folded (2x depth/noise).
    """
    if scale_factor == 1.0:
        return circuit.copy()
    
    fraction_to_fold = (scale_factor - 1.0) / 2.0
    folded_circuit = circuit.copy_empty_like()
    gates = list(circuit.data)
    num_gates = len(gates)
    num_to_fold = int(round(num_gates * fraction_to_fold))
    
    fold_indices = set(np.linspace(0, num_gates - 1, num_to_fold, dtype=int)) if num_to_fold > 0 else set()
    
    for idx, instruction in enumerate(gates):
        folded_circuit.append(instruction)
        if idx in fold_indices:
            gate = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits
            folded_circuit.append(gate.inverse(), qubits, clbits)
            folded_circuit.append(gate, qubits, clbits)
            
    return folded_circuit

def extrapolate_zero_noise(lambdas, energies):
    """Fits Linear and Richardson (Quadratic) curves to extrapolate to lambda = 0."""
    lambdas = np.array(lambdas)
    energies = np.array(energies)
    
    # Linear fit: E(lambda) = a * lambda + b  ==> b is E(0)
    p_lin = np.polyfit(lambdas, energies, 1)
    e_zne_linear = np.polyval(p_lin, 0.0)
    
    # Richardson / Quadratic fit: E(lambda) = a * lambda^2 + b * lambda + c ==> c is E(0)
    p_quad = np.polyfit(lambdas, energies, 2)
    e_zne_richardson = np.polyval(p_quad, 0.0)
    
    return e_zne_linear, e_zne_richardson

def run_day4_zne():
    # 1. Molecular Setup
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    qubit_hamiltonian = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    nuclear_repulsion = problem.nuclear_repulsion_energy

    # 2. Decomposed Hardware-Efficient Ansatz
    ansatz = TwoLocal(
        num_qubits=qubit_hamiltonian.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        insert_barriers=True
    ).decompose()

    # 3. Perform Noisy VQE to find optimal circuit parameters theta*
    noise_model = build_custom_noise_model()
    estimator = AerEstimator(
        options={"backend_options": {"noise_model": noise_model}}
    )
    optimizer = COBYLA(maxiter=120)

    print("=== STEP 1: RUNNING NOISY VQE OPTIMIZATION ===")
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    vqe_result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)
    
    noisy_unmitigated_energy = vqe_result.eigenvalue.real + nuclear_repulsion
    optimal_parameters = vqe_result.optimal_point
    
    print(f"Noisy Unmitigated Energy (lambda=1.0): {noisy_unmitigated_energy:.6f} Hartree\n")

    # 4. Bind optimal parameters into state preparation circuit
    optimal_circuit = ansatz.assign_parameters(optimal_parameters)

    # 5. Evaluate Circuit at Scaled Noise Levels (lambda = 1.0, 2.0, 3.0) using Gate Folding
    print("=== STEP 2: EVALUATING FOLDED CIRCUITS (GATE FOLDING ZNE) ===")
    scale_factors = [1.0, 2.0, 3.0]
    scaled_energies = []

    for scale in scale_factors:
        folded_circ = fold_circuit_global(optimal_circuit, scale_factor=scale)
        pub = (folded_circ, qubit_hamiltonian)
        job = estimator.run([pub])
        result = job.result()[0]
        e_val = float(result.data.evs) + nuclear_repulsion
        scaled_energies.append(e_val)
        print(f"Noise Scale Factor lambda = {scale:.1f} | Measured Energy: {e_val:.6f} Hartree")

    # 6. Extrapolate to Zero-Noise Limit (lambda = 0.0)
    e_zne_lin, e_zne_rich = extrapolate_zero_noise(scale_factors, scaled_energies)

    # 7. Comparative Analysis
    exact_fci = -1.137306
    ideal_vqe = -1.116990
    
    err_noisy = abs(noisy_unmitigated_energy - exact_fci) * 1000
    err_linear = abs(e_zne_lin - exact_fci) * 1000
    err_richardson = abs(e_zne_rich - exact_fci) * 1000

    print("\n==========================================================================")
    print("               DAY 4: ZERO-NOISE EXTRAPOLATION (ZNE) RESULTS              ")
    print("==========================================================================")
    print(f"Exact Classical (FCI) Target : {exact_fci:.6f} Hartree")
    print(f"Ideal Noise-Free VQE Baseline: {ideal_vqe:.6f} Hartree")
    print("-" * 74)
    print(f"1. Noisy Unmitigated (lambda=1.0): {noisy_unmitigated_energy:.6f} Hartree | Error: {err_noisy:.2f} mHa")
    print(f"2. ZNE Mitigated (Linear)       : {e_zne_lin:.6f} Hartree | Error: {err_linear:.2f} mHa")
    print(f"3. ZNE Mitigated (Richardson)   : {e_zne_rich:.6f} Hartree | Error: {err_richardson:.2f} mHa")
    print("==========================================================================")
    
    noise_recovered = ((noisy_unmitigated_energy - e_zne_rich) / 
                       (noisy_unmitigated_energy - ideal_vqe)) * 100
    print(f"Noise Shift Error Recovered by ZNE: {noise_recovered:.1f}%\n")

if __name__ == "__main__":
    run_day4_zne()
