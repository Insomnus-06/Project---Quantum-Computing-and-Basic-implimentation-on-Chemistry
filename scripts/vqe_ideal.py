"""
vqe_ideal.py
Day 2 Script: Run Variational Quantum Eigensolver (VQE) on an ideal simulator
and compare convergence against the classical FCI baseline (-1.137306 Hartree).
"""

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import VQE
from qiskit.primitives import StatevectorEstimator as Estimator

def run_vqe_ideal():
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

    # 2. Build Hardware-Efficient Ansatz (HEA)
    # 4 qubits using Ry rotation blocks and CZ entangling gates
    ansatz = TwoLocal(
        num_qubits=qubit_hamiltonian.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="linear",
        reps=1,
        insert_barriers=True
    )

    # 3. Callback function to log optimization progress
    def logging_callback(eval_count, parameters, mean, std_dev):
        if eval_count % 10 == 0 or eval_count == 1:
            current_total = mean + problem.nuclear_repulsion_energy
            print(f"Iteration {eval_count:3d}: Energy = {current_total:.6f} Hartree")

    # 4. Initialize Estimator primitive and Classical Optimizer
    estimator = Estimator()
    optimizer = COBYLA(maxiter=150)

    # 5. Execute VQE Loop
    print("=== STARTING IDEAL VQE OPTIMIZATION ===")
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=logging_callback
    )
    result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)

    # 6. Process Results
    nuclear_repulsion = problem.nuclear_repulsion_energy
    vqe_total_energy = result.eigenvalue.real + nuclear_repulsion
    exact_fci_energy = -1.137306

    print("\n=== VQE IDEAL RESULTS ===")
    print(f"VQE Ground State Energy : {vqe_total_energy:.6f} Hartree")
    print(f"Exact Classical (FCI)   : {exact_fci_energy:.6f} Hartree")
    print(f"Absolute Energy Error   : {abs(vqe_total_energy - exact_fci_energy):.6f} Hartree")
    print(f"Optimizer Iterations    : {result.cost_function_evals}")

if __name__ == "__main__":
    run_vqe_ideal()
