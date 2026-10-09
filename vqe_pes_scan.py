"""
vqe_pes_scan.py
Day 5 Script: Potential Energy Surface (PES) Scan for H2 Molecule.
Computes classical FCI, ideal VQE, noisy unmitigated VQE, and linear ZNE 
mitigated VQE across interatomic distances r in [0.3, 2.5] Angstroms.
Generates publication-quality plot 'h2_dissociation_curve.png'.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE, NumPyMinimumEigensolver

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
    """Folds circuit gates (g -> g g_dagger g) to scale noise by scale_factor lambda."""
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

def run_pes_scan():
    # Define bond distances (Angstroms) across dissociation curve
    distances = [0.3, 0.5, 0.7, 0.735, 0.9, 1.1, 1.4, 1.8, 2.2, 2.5]
    
    fci_energies = []
    ideal_energies = []
    noisy_energies = []
    zne_energies = []

    noise_model = build_custom_noise_model()
    
    print("==========================================================================")
    print("      DAY 5: H2 POTENTIAL ENERGY SURFACE (PES) SCAN & ZNE MITIGATION      ")
    print("==========================================================================")
    print(f"{'r (Angstrom)':<14} | {'FCI (Hartree)':<14} | {'Ideal VQE':<14} | {'Noisy VQE':<14} | {'ZNE Linear':<14}")
    print("-" * 78)

    for r in distances:
        # 1. Build molecular problem at bond distance r
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {r}",
            basis="sto-3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM
        )
        problem = driver.run()
        qubit_hamiltonian = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
        nuclear_repulsion = problem.nuclear_repulsion_energy

        # 2. Compute Exact Classical FCI Energy
        fci_solver = NumPyMinimumEigensolver()
        fci_result = fci_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
        e_fci = fci_result.eigenvalue.real + nuclear_repulsion
        fci_energies.append(e_fci)

        # 3. Setup Hardware-Efficient Ansatz
        ansatz = TwoLocal(
            num_qubits=qubit_hamiltonian.num_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            entanglement="linear",
            reps=1,
            insert_barriers=True
        ).decompose()

        optimizer = COBYLA(maxiter=120)

        # 4. Ideal Noise-Free VQE Simulation
        ideal_estimator = AerEstimator()
        vqe_ideal = VQE(estimator=ideal_estimator, ansatz=ansatz, optimizer=optimizer)
        res_ideal = vqe_ideal.compute_minimum_eigenvalue(qubit_hamiltonian)
        e_ideal = res_ideal.eigenvalue.real + nuclear_repulsion
        ideal_energies.append(e_ideal)

        # 5. Noisy VQE Optimization
        noisy_estimator = AerEstimator(
            options={"backend_options": {"noise_model": noise_model}}
        )
        vqe_noisy = VQE(estimator=noisy_estimator, ansatz=ansatz, optimizer=optimizer)
        res_noisy = vqe_noisy.compute_minimum_eigenvalue(qubit_hamiltonian)
        e_noisy = res_noisy.eigenvalue.real + nuclear_repulsion
        noisy_energies.append(e_noisy)

        # 6. Global Gate Folding ZNE (Linear)
        opt_circuit = ansatz.assign_parameters(res_noisy.optimal_point)
        scale_factors = [1.0, 2.0, 3.0]
        scaled_evals = []

        for scale in scale_factors:
            folded_circ = fold_circuit_global(opt_circuit, scale_factor=scale)
            pub = (folded_circ, qubit_hamiltonian)
            job = noisy_estimator.run([pub])
            eval_val = float(job.result()[0].data.evs) + nuclear_repulsion
            scaled_evals.append(eval_val)

        # Linear ZNE extrapolation to lambda = 0.0
        p_lin = np.polyfit(scale_factors, scaled_evals, 1)
        e_zne = np.polyval(p_lin, 0.0)
        zne_energies.append(e_zne)

        print(f"{r:<14.3f} | {e_fci:<14.6f} | {e_ideal:<14.6f} | {e_noisy:<14.6f} | {e_zne:<14.6f}")

    print("==========================================================================\n")

    # 7. Generate Dissociation Plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, fci_energies, 'k-', label='Exact Classical FCI Target', linewidth=2.0)
    plt.plot(distances, ideal_energies, 'g--', label='Ideal Noise-Free VQE', linewidth=1.5)
    plt.plot(distances, noisy_energies, 'r:', label=r'Noisy Unmitigated VQE ($\lambda=1.0$)', linewidth=1.5)
    plt.plot(distances, zne_energies, 'b-o', label='ZNE Mitigated VQE (Linear)', linewidth=2.0, markersize=5)

    plt.xlabel(r'Interatomic Distance $r$ ($\AA$)', fontsize=12)
    plt.ylabel('Ground State Energy (Hartree)', fontsize=12)
    plt.title(r'$H_2$ Potential Energy Surface (PES) & Zero-Noise Extrapolation', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('h2_dissociation_curve.png', dpi=300)
    print("Plot successfully generated and saved to 'h2_dissociation_curve.png'.\n")

if __name__ == "__main__":
    run_pes_scan()
