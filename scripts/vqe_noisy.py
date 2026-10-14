"""
vqe_noisy.py
Day 3 Script: Run Variational Quantum Eigensolver (VQE) under a realistic 
NISQ noise model (1-qubit/2-qubit gate errors & readout errors).
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

def build_custom_noise_model():
    """Constructs a realistic NISQ noise model."""
    noise_model = NoiseModel()
    
    # 1. Depolarizing error on single-qubit gates (0.5% error rate)
    p_1q = 0.005
    err_1q = depolarizing_error(p_1q, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ["ry", "rz", "x", "sx"])
    
    # 2. Depolarizing error on two-qubit CZ gates (2.0% error rate)
    p_2q = 0.02
    err_2q = depolarizing_error(p_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cz"])
    
    # 3. Readout (measurement) errors (2.0% bit-flip probability)
    p_read = 0.02
    readout_err = ReadoutError([[1 - p_read, p_read], [p_read, 1 - p_read]])
    noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model

def run_vqe_noisy():
    # 1. Define Molecule and Map Hamiltonian
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    qubit_hamiltonian = JordanWignerMapper().map(problem.hamiltonian.second_q_op())

    # 2. Build Hardware-Efficient Ansatz (HEA) and decompose into basic gates
    ansatz = TwoLocal(
        num_qubits=qubit_hamiltonian.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        insert_barriers=True
    ).decompose()

    # 3. Callback logging function
    def logging_callback(eval_count, parameters, mean, std_dev=None):
        if eval_count % 10 == 0 or eval_count == 1:
            current_total = mean + problem.nuclear_repulsion_energy
            print(f"Iteration {eval_count:3d}: Noisy Energy = {current_total:.6f} Hartree")

    # 4. Initialize EstimatorV2 Primitive with noise_model
    noise_model = build_custom_noise_model()
    estimator = AerEstimator(
        options={
            "backend_options": {
                "noise_model": noise_model
            }
        }
    )
    optimizer = COBYLA(maxiter=150)

    # 5. Execute Noisy VQE Loop
    print("=== STARTING NOISY VQE OPTIMIZATION (NISQ SIMULATION) ===")
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=logging_callback
    )
    result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)

    # 6. Process Results
    nuclear_repulsion = problem.nuclear_repulsion_energy
    vqe_noisy_energy = result.eigenvalue.real + nuclear_repulsion
    exact_fci_energy = -1.137306
    vqe_ideal_energy = -1.116990

    print("\n=== VQE NOISY SIMULATION RESULTS ===")
    print(f"Noisy VQE Energy       : {vqe_noisy_energy:.6f} Hartree")
    print(f"Ideal VQE Energy       : {vqe_ideal_energy:.6f} Hartree")
    print(f"Exact Classical (FCI)  : {exact_fci_energy:.6f} Hartree")
    print(f"Noise Error Shift      : {abs(vqe_noisy_energy - vqe_ideal_energy):.6f} Hartree")
    print(f"Total Baseline Error   : {abs(vqe_noisy_energy - exact_fci_energy):.6f} Hartree")

if __name__ == "__main__":
    run_vqe_noisy()
