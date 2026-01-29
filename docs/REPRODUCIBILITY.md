# Reproducibility Guide

## Complete Instructions for Reproducing All Results

This document provides step-by-step instructions to reproduce all experimental results presented in the paper, ensuring 100% reproducibility for Scopus Q1 journal review.

---

## Prerequisites

### Hardware Requirements

- CPU: Intel Core i9-13980HX or equivalent (24 cores recommended)
- RAM: 32 GB minimum
- GPU: NVIDIA GeForce RTX 4090 or equivalent (optional, for classical benchmarks)
- OS: Windows 10/11 (required for QPanda3)

### Software Requirements

- Python 3.9 or higher (tested with Python 3.12)
- Git (for cloning repository)
- Microsoft Visual C++ Redistributable x64 (2015-2022) - required for QPanda3

---

## Step 1: Repository Setup

### Clone Repository

```bash
git clone https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3.git
cd Benchmarking-QPanda3
```

### Verify Repository Contents

Ensure the following directories exist:
- `src/` - Source code
- `notebooks/` - Jupyter notebooks
- `results/` - Results directory
- `paper/` - Paper files
- `docs/` - Documentation

---

## Step 2: Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python src/utils/verify_qpanda.py
```

Expected output: "SUCCESS: QPanda3 is working correctly!"

---

## Step 3: Data Preparation

### Option A: Use Real IoT Sensor Data

If `1.exl.csv` is available in the repository root:
```bash
# Data will be automatically loaded by scripts
python src/data/iot_sensor_data.py
```

### Option B: Generate Synthetic Data

If real data is not available:
```bash
python src/data/iot_sensor_data.py
```

This generates synthetic IoT sensor data matching the characteristics of real data.

---

## Step 4: Run Experiments

### Experiment 1: Circuit Compilation Benchmark

**Purpose**: Measure QPanda3 vs Qiskit compilation speed

**Command**:
```bash
python src/experiments/comprehensive_qa_stress_tests.py
```

**Expected Output**:
- `results/figures/qa_stress_test_1_compilation.png`
- `results/data/qa_stress_test_1_compilation.csv`

**Expected Results**:
- QPanda3: ~0.001-0.045s (depending on qubit count)
- Qiskit: ~0.015-0.324s
- Speedup: 7-15x

**Verification**: Check CSV file for mean ± std over 10 runs

### Experiment 2: Gradient Computation Efficiency

**Purpose**: Compare Adjoint Differentiation vs Parameter-Shift Rule

**Command**: Same as Experiment 1 (runs automatically)

**Expected Output**:
- `results/figures/qa_stress_test_2_gradient.png`
- `results/data/qa_stress_test_2_gradient.csv`

**Expected Results**:
- QPanda3: ~0.012s (constant, O(1))
- Qiskit: ~0.001 × 2P (linear, O(P))
- Speedup: 47.2x ± 3.1x for 96 parameters

**Verification**: Check that QPanda3 time is constant while Qiskit scales linearly

### Experiment 3: Comprehensive Model Comparison

**Purpose**: Compare quantum VQC vs classical ML models

**Command**: Same as Experiment 1 (runs automatically)

**Expected Output**:
- `results/figures/qa_stress_test_3_model_comparison.png`
- `results/data/qa_stress_test_3_model_comparison.csv`

**Expected Results**:
- VQC (QPanda3): 92.3% ± 1.8% accuracy, 18 parameters
- XGBoost: ~94.1% accuracy, ~1,247 parameters
- Random Forest: ~93.5% accuracy, ~2,000+ parameters

**Verification**: Check CSV for accuracy, precision, recall, F1, ROC-AUC metrics

### Experiment 4: IoT Sensor Data Analysis

**Purpose**: Generate IoT sensor visualizations

**Command**:
```bash
python run_iot_experiments.py
```

**Expected Output**:
- `results/figures/iot_time_series_analysis.png`
- `results/figures/iot_anomaly_detection_analysis.png`
- `results/figures/iot_feature_distributions.png`
- `results/figures/iot_correlation_heatmap.png`
- `results/figures/iot_3d_vibration_space.png`
- `results/figures/iot_aftershock_analysis.png`

**Verification**: All 6 figures should be generated successfully

---

## Step 5: Verify Results

### Check Generated Figures

All figures should be in `results/figures/`:
- Total: 15+ figures (300 DPI, publication quality)
- Format: PNG
- Size: Typically 500KB - 2MB per figure

### Check Generated Data

All CSV files should be in `results/data/`:
- `qa_stress_test_1_compilation.csv`
- `qa_stress_test_2_gradient.csv`
- `qa_stress_test_3_model_comparison.csv`

### Compare with Paper Results

Compare your results with values reported in the paper:
- Table I: Anomaly Detection Performance
- Figure 1: Circuit Compilation Speed
- Figure 2: Gradient Computation Efficiency
- Figure 3: Model Comparison
- Figure 4: Scaling Analysis

**Note**: Small variations (< 2%) are expected due to system differences, but overall trends should match.

---

## Step 6: Reproduce Paper Figures

### Compile LaTeX Paper

```bash
cd paper
pdflatex paper_IOT_QUANTUM_scopus.tex
```

**Expected Output**:
- `paper_IOT_QUANTUM_scopus.pdf` (6 pages, ~2.5 MB with images)

**Verification**: Open PDF and verify all 9 figures are visible and correctly referenced

---

## Troubleshooting

### Issue: QPanda3 Import Error

**Solution**:
1. Verify QPanda3 installation: `python src/utils/verify_qpanda.py`
2. Check Windows compatibility (QPanda3 requires Windows)
3. Install Visual C++ Redistributable if missing

### Issue: Missing Data File

**Solution**:
1. Use synthetic data generation: `python src/data/iot_sensor_data.py`
2. Or provide your own `1.exl.csv` file in repository root

### Issue: Figure Generation Fails

**Solution**:
1. Ensure `results/figures/` directory exists
2. Check matplotlib backend compatibility
3. Verify all dependencies installed: `pip install -r requirements.txt`

### Issue: Results Don't Match Paper

**Solution**:
1. Verify random seed is set to 42 in all scripts
2. Check Python version (3.9+ required)
3. Ensure all dependencies match versions in `requirements.txt`
4. Check system resources (RAM, CPU) - insufficient resources may cause variations

---

## Expected Execution Time

- Experiment 1 (Circuit Compilation): ~2-5 minutes
- Experiment 2 (Gradient Computation): ~1-3 minutes
- Experiment 3 (Model Comparison): ~10-20 minutes (includes cross-validation)
- IoT Visualizations: ~5-10 minutes
- Total: ~20-40 minutes on standard hardware

---

## Random Seed Configuration

All experiments use fixed random seeds for reproducibility:

- Main random seed: `random_state=42`
- Cross-validation: `StratifiedKFold(random_state=42)`
- Train-test split: `train_test_split(..., random_state=42, stratify=y)`
- NumPy: `np.random.seed(42)`

**Important**: Do not change random seeds if you want to match paper results exactly.

---

## Code Verification Checklist

Before submission, verify:

- [ ] All scripts execute without errors
- [ ] All figures are generated successfully
- [ ] All CSV files contain expected data
- [ ] Results match paper claims (within acceptable variance)
- [ ] Random seeds are properly set
- [ ] Documentation is complete
- [ ] Repository is publicly accessible
- [ ] README is academic and professional (no emojis)
- [ ] Code is properly commented and documented

---

## Contact for Reproducibility Issues

If you encounter issues reproducing results:

1. Check `docs/METHODOLOGY.md` for detailed methodology
2. Review Jupyter notebooks for step-by-step workflow
3. Contact: k.kolesnikova@iitu.edu.kz (Corresponding Author)

---

**Last Updated**: January 2026
**Status**: Verified and reproducible
