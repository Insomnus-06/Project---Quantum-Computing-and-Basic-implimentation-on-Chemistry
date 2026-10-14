import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


def create_ansatz(num_qubits, reps=1):
    """Creates a native parameterized ansatz circuit using standard Ry and CZ gates."""
    circuit = QuantumCircuit(num_qubits)
    param_count = num_qubits * (reps + 1)
    params = ParameterVector('θ', param_count)
    
    idx = 0
    # Initial Ry rotation layer
    for q in range(num_qubits):
        circuit.ry(params[idx], q)
        idx += 1
        
    for r in range(reps):
        # Linear CZ entanglement layer (if > 1 qubit)
        if num_qubits > 1:
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        # Subsequent Ry rotation layer
        for q in range(num_qubits):
            circuit.ry(params[idx], q)
            idx += 1
            
    return circuit


def run_tapering_benchmark():
    r = 0.735  # Equilibrium bond length
    print(f"=== Day 6: Z2 Symmetry Tapering Benchmark (H2 at r = {r} Angstrom) ===\n")

    # 1. Generate Electronic Structure Problem
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {r}",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    second_q_op = problem.hamiltonian.second_q_op()
    nuclear_repulsion = problem.nuclear_repulsion_energy

    # 2. Standard 4-Qubit Mapping (Untapered Jordan-Wigner)
    jw_mapper = JordanWignerMapper()
    hamiltonian_4q = jw_mapper.map(second_q_op)

    # 3. Tapered Mapping (Automatic Z2 Symmetry Reduction)
    tapered_mapper = problem.get_tapered_mapper(jw_mapper)
    hamiltonian_tapered = tapered_mapper.map(second_q_op)

    num_q_untapered = hamiltonian_4q.num_qubits
    num_q_tapered = hamiltonian_tapered.num_qubits

    print(f"Untapered Operator Qubits : {num_q_untapered}")
    print(f"Tapered Operator Qubits   : {num_q_tapered}")
    print(f"Number of Pauli Terms (4Q): {len(hamiltonian_4q)}")
    print(f"Number of Pauli Terms (Tapered): {len(hamiltonian_tapered)}\n")

    # 4. Exact Classical FCI Baseline
    fci_solver = NumPyMinimumEigensolver()
    fci_energy = fci_solver.compute_minimum_eigenvalue(hamiltonian_tapered).eigenvalue.real + nuclear_repulsion
    print(f"Exact Classical FCI Target Energy: {fci_energy:.6f} Hartree\n")

    # 5. Build Native Ansatz Circuits
    ansatz_4q = create_ansatz(num_q_untapered, reps=1)
    ansatz_tapered = create_ansatz(num_q_tapered, reps=1)

    estimator = AerEstimator()
    optimizer = COBYLA(maxiter=100)

    # 6. Benchmark 4-Qubit VQE Execution
    t0 = time.time()
    vqe_4q = VQE(estimator=estimator, ansatz=ansatz_4q, optimizer=optimizer)
    res_4q = vqe_4q.compute_minimum_eigenvalue(hamiltonian_4q)
    e_4q = res_4q.eigenvalue.real + nuclear_repulsion
    t_4q = time.time() - t0

    # 7. Benchmark Tapered VQE Execution
    t0 = time.time()
    vqe_tapered = VQE(estimator=estimator, ansatz=ansatz_tapered, optimizer=optimizer)
    res_tapered = vqe_tapered.compute_minimum_eigenvalue(hamiltonian_tapered)
    e_tapered = res_tapered.eigenvalue.real + nuclear_repulsion
    t_tapered = time.time() - t0

    # 8. Output Benchmark Comparison Table
    cz_4q = ansatz_4q.count_ops().get("cz", 0)
    cz_tapered = ansatz_tapered.count_ops().get("cz", 0)

    print("-" * 75)
    print(f"{'Metric':<28} | {'Untapered (4 Qubits)':<20} | {'Tapered (' + str(num_q_tapered) + ' Qubit)':<20}")
    print("-" * 75)
    print(f"{'Qubit Count':<28} | {num_q_untapered:<20} | {num_q_tapered:<20}")
    print(f"{'Ansatz Depth':<28} | {ansatz_4q.depth():<20} | {ansatz_tapered.depth():<20}")
    print(f"{'CZ Gate Count':<28} | {cz_4q:<20} | {cz_tapered:<20}")
    print(f"{'Optimized Energy (Hartree)':<28} | {e_4q:<20.6f} | {e_tapered:<20.6f}")
    print(f"{'Absolute Error (mHa)':<28} | {abs(e_4q - fci_energy)*1000:<20.2f} | {abs(e_tapered - fci_energy)*1000:<20.2f}")
    print(f"{'Execution Time (seconds)':<28} | {t_4q:<20.3f} | {t_tapered:<20.3f}")
    print("-" * 75)


if __name__ == "__main__":
    run_tapering_benchmark()
