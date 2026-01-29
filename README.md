# Benchmarking QPanda3: A High-Performance Chinese Quantum Computing Framework

This repository contains the source code, experimental scripts, and data for reproducing the results presented in the paper "Quantum Machine Learning for IoT-Based Structural Health Monitoring: A QPanda3 Framework Evaluation for Real-Time Anomaly Detection in Building Sensor Networks."

## Repository Information

**Repository URL**: https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3

**License**: MIT License

**Python Version**: 3.9 or higher

## Overview

This project presents the first comprehensive performance benchmark of QPanda3, a high-performance quantum programming framework developed by Origin Quantum (OriginQ), against industry-standard Qiskit for hybrid quantum-classical machine learning applications. The research focuses on IoT-based structural health monitoring using real-world sensor data collected from building monitoring systems in Almaty, Kazakhstan.

### Key Contributions

- First comprehensive benchmark of QPanda3 for IoT applications
- Demonstration of 7-15x speedup in circuit compilation compared to Qiskit
- 47.2x ± 3.1x gradient computation speedup for deep circuits
- Quantum anomaly detection achieving 92.3% ± 1.8% accuracy with only 18 parameters
- Statistical validation of quantum advantage in parameter efficiency (p < 0.001)

## Project Structure

```
Benchmarking-QPanda3/
├── README.md
├── requirements.txt
├── LICENSE
├── setup.py
├── main.py
│
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── iot_sensor_data.py      # IoT sensor data generator
│   ├── models/
│   │   ├── vqc.py                  # Variational Quantum Classifier
│   │   └── classical.py            # Classical baseline models
│   ├── experiments/
│   │   ├── benchmark_stress_test.py
│   │   ├── comprehensive_qa_stress_tests.py
│   │   ├── iot_quantum_anomaly_detection.py
│   │   ├── generate_iot_visualizations.py
│   │   ├── run_vqc_experiment.py
│   │   └── run_comprehensive_experiments.py
│   └── utils/
│       └── verify_qpanda.py        # QPanda3 installation verification
│
├── notebooks/
│   ├── 01_Data_Exploration_and_Analysis.ipynb
│   ├── 02_QA_Stress_Test_Experiments.ipynb
│   └── 03_Quantum_Anomaly_Detection.ipynb
│
├── results/
│   ├── figures/                    # All experimental plots (300 DPI)
│   └── data/                       # CSV files with raw results
│
├── paper/
│   ├── paper_IOT_QUANTUM_scopus.tex
│   └── paper_IOT_QUANTUM_scopus.pdf
│
└── docs/
    ├── METHODOLOGY.md              # Detailed experimental methodology
    ├── REPRODUCIBILITY.md         # Complete reproducibility guide
    ├── CITATIONS_ADDED.md         # Citation documentation
    └── FIGURES_IN_PAPER.md        # Figure verification
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Windows 10/11 (required for QPanda3)
- Microsoft Visual C++ Redistributable x64 (2015-2022)
- Git (for cloning repository)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3.git
cd Benchmarking-QPanda3
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify QPanda3 installation:
```bash
python src/utils/verify_qpanda.py
```

## Usage

### Running QA Stress Tests

Execute comprehensive QA stress test suite:
```bash
python src/experiments/comprehensive_qa_stress_tests.py
```

This script performs:
- QA Stress Test 1: Circuit compilation benchmark (QPanda3 vs Qiskit)
- QA Stress Test 2: Gradient computation efficiency (Adjoint vs Parameter-Shift)
- QA Stress Test 3: Comprehensive model comparison (quantum vs classical)

### Running IoT Experiments

Generate IoT sensor visualizations:
```bash
python run_iot_experiments.py
```

Run quantum anomaly detection:
```bash
python src/experiments/iot_quantum_anomaly_detection.py
```

### Using Jupyter Notebooks

Launch Jupyter to explore the research workflow:
```bash
jupyter notebook notebooks/
```

Notebooks provide step-by-step methodology:
- Data exploration and analysis
- QA stress test experiments
- Quantum anomaly detection training and evaluation

## Experiments

### Experiment 1: Circuit Compilation Speed Benchmark

**Purpose**: Evaluate QPanda3 compilation efficiency vs Qiskit

**Configuration**: Circuits with 100, 500, 1000, and 2000 qubits

**Statistical Rigor**: 10 independent runs per configuration, mean ± std reported

**Result**: QPanda3 achieves 7-15x speedup over Qiskit

**Script**: `src/experiments/comprehensive_qa_stress_tests.py` (experiment_1_circuit_compilation_benchmark)

### Experiment 2: Gradient Computation Efficiency

**Purpose**: Compare Adjoint Differentiation (QPanda3) vs Parameter-Shift Rule (Qiskit)

**Mathematical Background**: 
- Adjoint Differentiation: O(1) complexity, constant time
- Parameter-Shift Rule: O(P) complexity, requires 2P circuit evaluations

**Configuration**: 6 qubits, 2-16 layers (12-96 parameters)

**Result**: QPanda3 achieves 47.2x ± 3.1x speedup for deep circuits

**Script**: `src/experiments/comprehensive_qa_stress_tests.py` (experiment_2_gradient_computation_benchmark)

### Experiment 3: Comprehensive Model Comparison

**Purpose**: Compare quantum VQC vs classical ML models

**Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC

**Statistical Rigor**: 5-fold cross-validation, 10 independent runs

**Models Evaluated**: VQC (QPanda3), XGBoost, Random Forest, SVM, MLP, Decision Tree

**Result**: VQC achieves 92.3% ± 1.8% accuracy with only 18 parameters vs 100-2000+ for classical models

**Script**: `src/experiments/comprehensive_qa_stress_tests.py` (experiment_3_model_comparison_comprehensive)

### Experiment 4: IoT Sensor Data Analysis

**Purpose**: Analyze real-world IoT sensor data for anomaly detection

**Dataset**: 258,463 sensor readings from building monitoring systems, Almaty, Kazakhstan

**Features**: Vibration sensors (X, Y, Z), environmental sensors (Temperature, Humidity, Pressure), aftershock detection

**Script**: `src/experiments/iot_quantum_anomaly_detection.py`

## Results

All experimental results are stored in the `results/` directory:

- `results/figures/` - All plots in PNG format (300 DPI, publication quality)
- `results/data/` - CSV files with raw numerical results

### Key Findings

- Circuit Compilation: QPanda3 achieves 7-15x speedup over Qiskit (p < 0.001)
- Gradient Computation: 47.2x ± 3.1x speedup for 96-parameter circuits
- Anomaly Detection: VQC achieves 92.3% ± 1.8% accuracy with 18 parameters
- Parameter Efficiency: 99% reduction compared to classical models

## Dataset

### IoT Sensor Data

**Source**: Building monitoring systems, Almaty, Kazakhstan

**Size**: 258,463 sensor readings

**Features**:
- Vibration sensors: X, Y, Z accelerometer readings
- Aftershock detection: Binary indicators for seismic events
- Environmental sensors: Temperature (°C), Humidity (%), Pressure (hPa)
- Derived features: Vibration magnitude, variance

**Sampling**: 10-second intervals

**Class Distribution**: ~3-5% anomalies (realistic for structural health monitoring)

**Data Availability**: Real-world data file (`1.exl.csv`) or synthetic data generation via `src/data/iot_sensor_data.py`

## Reproducibility

### Random Seeds

All experiments use fixed random seeds for reproducibility:
- Main random seed: 42
- Cross-validation: StratifiedKFold with random_state=42
- Train-test split: random_state=42, stratify=y

### Environment

Hardware specifications documented in paper:
- CPU: Intel Core i9-13980HX (24 cores, 32 threads)
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16 GB VRAM)
- RAM: 32 GB DDR5
- OS: Windows 11 Pro

Software versions:
- Python 3.12
- pyqpanda3 0.3.2
- Qiskit 2.3.0
- scikit-learn 1.3.0
- NumPy 1.26.0
- Pandas 2.0.0

### Reproducing Results

1. Follow installation instructions above
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Run experiments in order:
   ```bash
   python src/experiments/comprehensive_qa_stress_tests.py
   python run_iot_experiments.py
   ```
4. Verify results match paper claims (see `docs/METHODOLOGY.md` for expected results)

## Paper

The complete research paper is available in the `paper/` directory:

- LaTeX source: `paper_IOT_QUANTUM_scopus.tex`
- PDF: `paper_IOT_QUANTUM_scopus.pdf`

**Title**: "Quantum Machine Learning for IoT-Based Structural Health Monitoring: A QPanda3 Framework Evaluation for Real-Time Anomaly Detection in Building Sensor Networks"

**Authors**:
- Nurkamila A. Daurenbayeva (IITU) - First Author
- Syrym Zhakypbekov (IITU)
- Artem A. Bykov (IITU)
- Kateryna V. Kolesnikova (IITU) - Corresponding Author

**Affiliation**: International IT University (IITU), Almaty, Kazakhstan

## Code Availability

All source code is publicly available at:
https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3

The repository includes:
- Complete source code for all experiments
- Data loading and preprocessing scripts
- Model implementations (quantum and classical)
- Jupyter notebooks documenting research workflow
- Comprehensive documentation

## Documentation

Detailed methodology and reproducibility instructions:
- `docs/METHODOLOGY.md` - Complete experimental methodology
- `docs/REPRODUCIBILITY.md` - Step-by-step reproduction guide
- `docs/CITATIONS_ADDED.md` - Citation documentation
- `docs/FIGURES_IN_PAPER.md` - Figure verification

## Citation

If you use this work in your research, please cite:

```bibtex
@article{daurenbayeva2025quantum,
  title={Quantum Machine Learning for IoT-Based Structural Health Monitoring: A QPanda3 Framework Evaluation for Real-Time Anomaly Detection in Building Sensor Networks},
  author={Daurenbayeva, Nurkamila A. and Zhakypbekov, Syrym and Bykov, Artem A. and Kolesnikova, Kateryna V.},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]},
  note={Code available at: https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3}
}
```

## Authors

- Nurkamila A. Daurenbayeva - First Author
  - Email: n.daurenbayeva@edu.iitu.kz
  - Affiliation: International IT University (IITU), Almaty, Kazakhstan

- Syrym Zhakypbekov
  - Email: s.zhakypbekov@iitu.edu.kz
  - Affiliation: International IT University (IITU), Almaty, Kazakhstan

- Artem A. Bykov
  - Email: a.bykov@edu.iitu.kz
  - Affiliation: International IT University (IITU), Almaty, Kazakhstan

- Kateryna V. Kolesnikova - Corresponding Author
  - Email: k.kolesnikova@iitu.edu.kz
  - Affiliation: International IT University (IITU), Almaty, Kazakhstan

## Acknowledgments

- Origin Quantum (OriginQ) for providing the QPanda3 framework and technical support
- International IT University (IITU) for computational resources and research infrastructure
- Building monitoring systems in Almaty, Kazakhstan for providing IoT sensor data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions regarding the code, experiments, or paper, please contact:
- Corresponding Author: Kateryna V. Kolesnikova (k.kolesnikova@iitu.edu.kz)
- Repository: https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3

---

Last Updated: January 2026
