import time
import numpy as np
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


def run_qeom_visualization():
    r = 0.735  # Equilibrium bond length
    print(f"=== Day 8: Generating qEOM Spectroscopic Energy Level Plot (H2 at r = {r} Å) ===\n")

    # 1. Molecular Setup & Problem Definition
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {r}",
        basis="sto-3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
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
    print("Running qEOM solver...")
    calculation_result = qeom_solver.solve(problem)

    # 4. Extract Energy Data
    eigenvalues = calculation_result.eigenvalues
    ground_energy = eigenvalues[0]
    excited_energies = eigenvalues[1:]
    
    state_labels = ['Ground ($E_0$)'] + [f'Excited ($E_{i+1}$)' for i in range(len(excited_energies))]
    all_energies = [ground_energy] + list(excited_energies)
    gaps = [e - ground_energy for e in all_energies]

    # 5. Generate Professional Matplotlib Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Energy Level Diagram ---
    for i, (label, energy) in enumerate(zip(state_labels, all_energies)):
        ax1.hlines(energy, xmin=0.2, xmax=0.8, color='royalblue' if i == 0 else 'darkorange', linewidth=3)
        ax1.text(0.82, energy, f'  {label}\n  {energy:.4f} Ha', va='center', fontsize=10, fontweight='bold')

    ax1.set_xlim(0, 1.2)
    ax1.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
    ax1.set_title('H2 qEOM Electronic Energy Levels', fontsize=13, fontweight='bold')
    ax1.set_xticks([])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Panel 2: Optical Excitation Gaps (ΔE) ---
    bars = ax2.bar(state_labels[1:], gaps[1:], color='forestgreen', width=0.5, edgecolor='black')
    ax2.set_ylabel('Excitation Gap $\Delta E$ from Ground State (Hartree)', fontsize=12, fontweight='bold')
    ax2.set_title('Optical Excitation Gaps ($\Delta E = E_n - E_0$)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f} Ha',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('qeom_spectrum.png', dpi=300)
    print("\nSuccessfully generated and saved plot as 'qeom_spectrum.png'!")


if __name__ == "__main__":
    run_qeom_visualization()
