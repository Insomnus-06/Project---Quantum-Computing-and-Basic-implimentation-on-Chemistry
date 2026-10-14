"""
vqe_noise_sweep.py
Day 3 Advanced Analysis: 
1. Noise Scaling Sweep (lambda = 0.0 to 2.0)
2. Noise Channel Isolation (Gate Noise vs Readout Error)
"""

import warnings
warnings.filterwarnings("ignore")

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE

# Qiskit Aer Primitives and Noise Imports
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

def build_noise_model(scale_factor=1.0, gate_noise=True, readout_noise=True):
    """Constructs a custom noise model scaled by factor lambda (scale_factor)."""
    noise_model = NoiseModel()
    
    if gate_noise and scale_factor > 0:
        p_1q = min(0.005 * scale_factor, 0.99)
        err_1q = depolarizing_error(p_1q, 1)
        noise_model.add_all_qubit_quantum_error(err_1q, ["ry", "rz", "x", "sx"])
        
        p_2q = min(0.02 * scale_factor, 0.99)
        err_2q = depolarizing_error(p_2q, 2)
        noise_model.add_all_qubit_quantum_error(err_2q, ["cz"])
        
    if readout_noise and scale_factor > 0:
        p_read = min(0.02 * scale_factor, 0.49)
        readout_err = ReadoutError([[1 - p_read, p_read], [p_read, 1 - p_read]])
        noise_model.add_all_qubit_readout_error(readout_err)
        
    return noise_model

def evaluate_vqe(qubit_hamiltonian, ansatz, nuclear_repulsion, noise_model):
    """Executes VQE with a specific noise model and returns total energy."""
    estimator = AerEstimator(
        options={"backend_options": {"noise_model": noise_model}}
    )
    optimizer = COBYLA(maxiter=120)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)
    return result.eigenvalue.real + nuclear_repulsion

def run_sweep():
    # 1. Setup Molecule & Qubit Hamiltonian
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
    exact_fci = -1.137306

    # 2. Decomposed Ansatz
    ansatz = TwoLocal(
        num_qubits=qubit_hamiltonian.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        insert_barriers=True
    ).decompose()

    print("\n==========================================================================")
    print(" PART 1: NOISE LEVEL SWEEP (Scaling Factor lambda: 0.0 -> 2.0)")
    print("==========================================================================")
    print(f"{'Scale Factor (lambda)':<22} | {'Measured Energy (Hartree)':<26} | {'Error vs FCI (mHa)':<20}")
    print("-" * 75)

    scales = [0.0, 0.5, 1.0, 1.5, 2.0]
    sweep_results = {}
    for scale in scales:
        nm = build_noise_model(scale_factor=scale, gate_noise=True, readout_noise=True)
        energy = evaluate_vqe(qubit_hamiltonian, ansatz, nuclear_repulsion, nm)
        sweep_results[scale] = energy
        err_mha = abs(energy - exact_fci) * 1000
        print(f"{scale:<22.1f} | {energy:<26.6f} | {err_mha:<20.2f}")

    print("\n==========================================================================")
    print(" PART 2: NOISE CHANNEL ISOLATION (At lambda = 1.0)")
    print("==========================================================================")
    
    # Gate Noise Only
    nm_gate_only = build_noise_model(scale_factor=1.0, gate_noise=True, readout_noise=False)
    e_gate_only = evaluate_vqe(qubit_hamiltonian, ansatz, nuclear_repulsion, nm_gate_only)

    # Readout Noise Only
    nm_read_only = build_noise_model(scale_factor=1.0, gate_noise=False, readout_noise=True)
    e_read_only = evaluate_vqe(qubit_hamiltonian, ansatz, nuclear_repulsion, nm_read_only)

    e_ideal = sweep_results[0.0]
    e_total = sweep_results[1.0]

    print(f"Ideal VQE Baseline (lambda=0.0) : {e_ideal:.6f} Hartree")
    print(f"Gate Noise Only                 : {e_gate_only:.6f} Hartree  (Shift: +{abs(e_gate_only - e_ideal)*1000:.2f} mHa)")
    print(f"Readout Noise Only              : {e_read_only:.6f} Hartree  (Shift: +{abs(e_read_only - e_ideal)*1000:.2f} mHa)")
    print(f"Combined Noise (Full NISQ)      : {e_total:.6f} Hartree  (Shift: +{abs(e_total - e_ideal)*1000:.2f} mHa)")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_sweep()
