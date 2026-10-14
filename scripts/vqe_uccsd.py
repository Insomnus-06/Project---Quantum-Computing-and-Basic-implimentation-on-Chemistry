import time
import numpy as np
from qiskit import transpile
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import EstimatorV2 as AerEstimator


def run_uccsd_benchmark():
    r = 0.735  # Equilibrium bond length
    print(f"=== Day 7: UCCSD Chemically Inspired Ansatz Benchmark (H2 at r = {r} Angstrom) ===\n")

    # 1. Generate Electronic Structure Problem via PySCF
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

    # 2. Map Fermionic Hamiltonian to Qubits (Jordan-Wigner)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(second_q_op)

    # 3. Exact Classical FCI Baseline
    fci_solver = NumPyMinimumEigensolver()
    fci_energy = fci_solver.compute_minimum_eigenvalue(qubit_op).eigenvalue.real + nuclear_repulsion
    print(f"Exact Classical FCI Target Energy: {fci_energy:.6f} Hartree\n")

    # 4. Build Hartree-Fock Initial State & UCCSD Ansatz
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles

    init_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    uccsd_ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=init_state
    )

    # Transpile to compile PauliEvolution into native gates for Aer
    uccsd_ansatz = transpile(
        uccsd_ansatz,
        basis_gates=['cx', 'rz', 'ry', 'x', 'sx', 'id'],
        optimization_level=1
    )

    print(f"UCCSD Ansatz Parameter Count: {uccsd_ansatz.num_parameters}")
    print(f"UCCSD Ansatz Qubit Count    : {uccsd_ansatz.num_qubits}\n")

    # 5. Configure VQE with Aer Estimator and COBYLA Optimizer
    estimator = AerEstimator()
    optimizer = COBYLA(maxiter=150)

    # 6. Execute UCCSD VQE
    print("Running UCCSD VQE Optimization...")
    t0 = time.time()
    vqe = VQE(estimator=estimator, ansatz=uccsd_ansatz, optimizer=optimizer)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    uccsd_energy = result.eigenvalue.real + nuclear_repulsion
    execution_time = time.time() - t0

    # 7. Output Results
    absolute_error = abs(uccsd_energy - fci_energy) * 1000  # in mHa

    print("\n" + "=" * 60)
    print(f"{'UCCSD Benchmark Metric':<30} | {'Value':<25}")
    print("=" * 60)
    print(f"{'Optimized Energy (Hartree)':<30} | {uccsd_energy:<25.6f}")
    print(f"{'Exact FCI Energy (Hartree)':<30} | {fci_energy:<25.6f}")
    print(f"{'Absolute Error (mHa)':<30} | {absolute_error:<25.2f}")
    print(f"{'Circuit Depth':<30} | {uccsd_ansatz.depth():<25}")
    print(f"{'Execution Time (seconds)':<30} | {execution_time:<25.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_uccsd_benchmark()
