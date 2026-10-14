# Quantum Computing , Quantum Chemistry and VQE Portfolio  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-v2.x-orange.svg)](https://qiskit.org/)
[![PySCF](https://img.shields.io/badge/PySCF-Quantum%20Chemistry-green.svg)](https://pyscf.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-grade computational quantum chemistry portfolio developed as a physics undergraduate project at **IIT Kharagpur**. This repository implements an end-to-end Variational Quantum Eigensolver (VQE) and Quantum Equation of Motion (qEOM) pipeline to study the hydrogen molecule (H2) across molecular dissociation, symmetry tapering, noise mitigation, excited-state spectroscopy, and optical transition properties.

---

## Repository Structure

```text
vqe_quantum_chemistry/
├── plots/
│   ├── h2_dissociation_curve.png
│   ├── qeom_spectrum.png
│   ├── excited_pes_scan.png
│   └── transition_spectrum.png
├── scripts/
│   ├── baseline_h2.py
│   ├── vqe_ideal.py
│   ├── vqe_noisy.py
│   ├── vqe_noise_sweep.py
│   ├── vqe_mitiq_zne.py
│   ├── vqe_pes_scan.py
│   ├── vqe_tapered.py
│   ├── vqe_uccsd.py
│   ├── vqe_qeom.py
│   ├── vqe_excited_pes.py
│   └── vqe_transition_dipoles.py
├── requirements.txt
└── README.md
