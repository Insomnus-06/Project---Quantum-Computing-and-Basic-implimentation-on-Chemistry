import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless / WSL environments
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_nature.second_q.algorithms import GroundStateEigensolver, QEOM


def run_full_excited_pes_scan():
    # Define bond length coordinate sweep from 0.5 Å to 1.5 Å
    r_values = np.linspace(0.5, 1.5, 9)
    
    # Storage lists for ground state and all 3 excited states
    ground_energies = []
    e1_energies = []
    e2_energies = []
    e3_energies = []

    print(f"=== Day 9: Full Multi-State Potential Energy Surface (PES) Scan for H2 ===\n")
    t_start = time.time()

    for r in r_values:
        print(f"Computing molecular geometry at bond length r = {r:.3f} Å...")
        
        # 1. Molecular Setup via PySCF
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {r}",
            basis="sto-3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM
        )
        problem = driver.run()
        nuclear_repulsion = problem.nuclear_repulsion_energy
        mapper = JordanWignerMapper()

        # 2. VQE Ground State Setup
        estimator = AerEstimator()
        optimizer = COBYLA(maxiter=150)
        init_state = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper
        )
        ansatz = UCCSD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=init_state
        )
        ansatz = transpile(ansatz, basis_gates=['cx', 'rz', 'ry', 'x', 'sx', 'id'], optimization_level=1)

        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
        gs_solver = GroundStateEigensolver(mapper, vqe)

        # 3. qEOM Solver Execution
        qeom_solver = QEOM(gs_solver, estimator, excitations="sd")
        calc_result = qeom_solver.solve(problem)

        # 4. Extract All Eigenvalues with Nuclear Repulsion Correction
        eigenvalues = calc_result.eigenvalues + nuclear_repulsion
        ground_energies.append(eigenvalues[0])
        e1_energies.append(eigenvalues[1])
        e2_energies.append(eigenvalues[2])
        e3_energies.append(eigenvalues[3])

    execution_time = time.time() - t_start
    print(f"\n Full PES scan completed in {execution_time:.2f} seconds.")

    # 5. Generate and Save Multi-Curve PES Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, ground_energies, 'bo-', linewidth=2, markersize=5, label='Ground State ($E_0$)')
    plt.plot(r_values, e1_energies, 'ro-', linewidth=2, markersize=5, label='Excited State ($E_1$)')
    plt.plot(r_values, e2_energies, 'go-', linewidth=2, markersize=5, label='Excited State ($E_2$)')
    plt.plot(r_values, e3_energies, 'mo-', linewidth=2, markersize=5, label='Excited State ($E_3$)')
    
    plt.xlabel('Bond Length $r$ (Angstrom)', fontsize=12, fontweight='bold')
    plt.ylabel('Total Molecular Energy (Hartree)', fontsize=12, fontweight='bold')
    plt.title('H2 Complete Multi-State Potential Energy Surface (qEOM-VQE)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig('excited_pes_scan.png', dpi=300)
    print("Successfully generated and saved multi-state PES plot as 'excited_pes_scan.png'!")


if __name__ == "__main__":
    run_full_excited_pes_scan()
