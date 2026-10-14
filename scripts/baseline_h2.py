from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver

def run_classical_baseline():
    # 1. Define H2 molecule at equilibrium bond length (0.735 Angstroms)
    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )

    # 2. Extract Electronic Structure Problem
    problem = driver.run()
    hamiltonian_second_q = problem.hamiltonian.second_q_op()

    # 3. Map Fermionic Operator to Qubit Operator via Jordan-Wigner Transformation
    mapper = JordanWignerMapper()
    qubit_hamiltonian = mapper.map(hamiltonian_second_q)

    print("=== 1. QUBIT HAMILTONIAN ===")
    print(f"Number of Qubits required: {qubit_hamiltonian.num_qubits}")
    print(f"Number of Pauli Terms: {len(qubit_hamiltonian)}")
    print("\nSample Pauli Terms:\n", qubit_hamiltonian[:3])

    # 4. Compute Exact Ground State Energy Classically (Diagonalization)
    numpy_solver = NumPyMinimumEigensolver()
    exact_result = numpy_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
    nuclear_repulsion = problem.nuclear_repulsion_energy
    total_exact_energy = exact_result.eigenvalue.real + nuclear_repulsion

    print("\n=== 2. CLASSICAL EXACT ENERGY (FCI) ===")
    print(f"Electronic Energy: {exact_result.eigenvalue.real:.6f} Hartree")
    print(f"Nuclear Repulsion: {nuclear_repulsion:.6f} Hartree")
    print(f"Total Ground State Energy: {total_exact_energy:.6f} Hartree")

if __name__ == "__main__":
    run_classical_baseline()

