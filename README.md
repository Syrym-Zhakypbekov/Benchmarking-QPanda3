# Benchmarking QPanda3: A High-Performance Chinese Quantum Computing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code and experiments for benchmarking QPanda3, a high-performance quantum programming framework developed by Origin Quantum (OriginQ), against industry-standard Qiskit for hybrid quantum-classical machine learning applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Paper](#paper)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🎯 Overview

This project presents the first comprehensive performance benchmark of QPanda3, demonstrating:
- **7-15× speedup** in circuit compilation compared to Qiskit
- **Orders of magnitude** improvement in gradient computation via Adjoint Differentiation
- **Competitive classification performance** (88.2% ± 1.3%) with only 12 parameters
- **Superior parameter efficiency** compared to classical ML models

## ✨ Features

- Comprehensive QA stress testing across multiple dimensions
- Scaling studies (4-10 qubits)
- Ansatz architecture comparisons
- Hyperparameter sensitivity analysis
- Statistical rigor (10 runs per experiment with mean ± std)
- Real-world medical diagnostics validation (Breast Cancer Wisconsin dataset)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Windows 10/11 (for QPanda3)
- Microsoft Visual C++ Redistributable x64 (2015-2022)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3.git
cd Benchmarking-QPanda3
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify QPanda3 installation:**
```bash
python src/utils/verify_qpanda.py
```

## 📁 Project Structure

```
Benchmarking-QPanda3/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── LICENSE                  # MIT License
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/              # VQC and classical models
│   │   ├── __init__.py
│   │   ├── vqc.py          # Variational Quantum Classifier
│   │   └── classical.py    # Classical baseline models
│   ├── experiments/        # Experiment scripts
│   │   ├── __init__.py
│   │   ├── benchmark_stress_test.py
│   │   ├── scaling_study.py
│   │   ├── ansatz_comparison.py
│   │   └── hyperparameter_analysis.py
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── verify_qpanda.py
│       └── visualization.py
│
├── notebooks/              # Jupyter notebooks for analysis
│   └── analysis.ipynb
│
├── results/                # Experimental results
│   ├── figures/           # Generated plots
│   └── data/              # CSV results
│
├── paper/                  # Paper-related files
│   ├── paper_ULTIMATE_scopus.tex
│   ├── paper_ULTIMATE_scopus.pdf
│   └── paper_for_scopus_ULTIMATE.docx
│
└── docs/                   # Documentation
    ├── BRUTAL_QA_AUDIT.md
    ├── FINAL_QA_ASSESSMENT.md
    └── PAPER_SUMMARY.md
```

## 💻 Usage

### Quick Start

Run the comprehensive benchmark suite:

```bash
python src/experiments/benchmark_stress_test.py
```

Train VQC on Breast Cancer dataset:

```bash
python src/experiments/run_vqc_experiment.py
```

Run all comprehensive experiments:

```bash
python src/experiments/run_comprehensive_experiments.py
```

### Individual Experiments

**1. Circuit Construction Benchmark:**
```bash
python src/experiments/benchmark_stress_test.py --experiment circuit
```

**2. Gradient Computation Benchmark:**
```bash
python src/experiments/benchmark_stress_test.py --experiment gradient
```

**3. Scaling Study:**
```bash
python src/experiments/scaling_study.py
```

**4. Ansatz Comparison:**
```bash
python src/experiments/ansatz_comparison.py
```

## 🔬 Experiments

### Experiment 1: Circuit Construction Speed
- **Qubit counts**: 100, 500, 1000, 2000
- **Comparison**: QPanda3 vs Qiskit
- **Result**: 7-15× speedup (mean ± std over 10 runs)

### Experiment 2: Gradient Computation Efficiency
- **Circuit depths**: 2, 4, 8, 16 layers
- **Comparison**: Adjoint Differentiation vs Parameter-Shift
- **Result**: Orders of magnitude speedup

### Experiment 3: Scaling Study
- **Qubit counts**: 4, 6, 8, 10
- **Metrics**: Accuracy, Parameters, Training Time
- **Result**: Accuracy improves 88.2% → 91.5%

### Experiment 4: Ansatz Architecture Comparison
- **Architectures**: HEA, RealAmplitudes, EfficientSU2
- **Metrics**: Parameters, Gates, Accuracy
- **Result**: HEA optimal (88.2% ± 1.3%)

### Experiment 5: Hyperparameter Sensitivity
- **Learning rates**: 0.01, 0.1, 0.5
- **Layers**: 1, 2, 3, 4, 5
- **Result**: Optimal LR=0.1, Layers=3

### Experiment 6: Classical Baseline Comparison
- **Models**: XGBoost, Random Forest, SVM, MLP, Decision Tree, VQC
- **Metrics**: Accuracy, Parameters, Training Time
- **Result**: VQC competitive with 12 params vs 100-2000+

## 📊 Results

All experimental results are stored in `results/` directory:
- `results/figures/` - Generated plots (PNG format, 300 DPI)
- `results/data/` - CSV files with raw results

Key findings:
- QPanda3 achieves **7-15× compilation speedup**
- **47.2× ± 3.1×** gradient computation speedup for deep circuits
- VQC achieves **88.2% ± 1.3%** accuracy with only **12 parameters**
- Statistical significance confirmed (p < 0.001)

## 📄 Paper

The complete research paper is available in the `paper/` directory:
- **LaTeX source**: `paper_ULTIMATE_scopus.tex`
- **PDF**: `paper_ULTIMATE_scopus.pdf`
- **DOCX**: `paper_for_scopus_ULTIMATE.docx`

**Title**: "Benchmarking QPanda3: A High-Performance Chinese Quantum Computing Framework for Hybrid Quantum-Classical Machine Learning on NISQ Devices"

**Authors**: 
- Nurkamila A. Daurenbayeva (IITU) - *First Author*
- Syrym Zhakypbekov (IITU)
- Artem A. Bykov (IITU)
- Kateryna V. Kolesnikova (IITU)

## 🛠️ Development

### Code Style

This project follows PEP 8 style guidelines. Use black for formatting:

```bash
pip install black
black src/
```

### Testing

Run verification script:

```bash
python src/utils/verify_qpanda.py
```

## 📝 Dataset

We use the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset:
- **Source**: University of Wisconsin-Madison Hospitals
- **Creator**: Dr. William H. Wolberg
- **Samples**: 569 (212 Malignant, 357 Benign)
- **Features**: 30 → 4 (via PCA, 95.2% variance preserved)
- **Access**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{zhakypbekov2025benchmarking,
  title={Benchmarking QPanda3: A High-Performance Chinese Quantum Computing Framework for Hybrid Quantum-Classical Machine Learning on NISQ Devices},
  author={Zhakypbekov, Syrym and Bykov, Artem A. and Daurenbayeva, Nurkamila A. and Kolesnikova, Kateryna V.},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]}
}
```

## 👥 Authors

- **Nurkamila A. Daurenbayeva** - *First Author* - [n.daurenbayeva@edu.iitu.kz](mailto:n.daurenbayeva@edu.iitu.kz)
- **Syrym Zhakypbekov** - [s.zhakypbekov@iitu.edu.kz](mailto:s.zhakypbekov@iitu.edu.kz)
- **Artem A. Bykov** - [a.bykov@edu.iitu.kz](mailto:a.bykov@edu.iitu.kz)
- **Kateryna V. Kolesnikova** - *Corresponding Author* - [k.kolesnikova@iitu.edu.kz](mailto:k.kolesnikova@iitu.edu.kz)

**Affiliation**: International IT University (IITU), Almaty, Kazakhstan

## 🙏 Acknowledgments

- Origin Quantum (OriginQ) for providing QPanda3 framework
- International IT University (IITU) for computational resources
- UCI Machine Learning Repository for the Breast Cancer dataset

## 📧 Contact

For questions or collaborations, please contact:
- **Syrym Zhakypbekov**: s.zhakypbekov@iitu.edu.kz
- **Project Repository**: https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3

---

**Last Updated**: January 2026
