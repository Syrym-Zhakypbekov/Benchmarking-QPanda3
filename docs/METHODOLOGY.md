# Comprehensive Methodology Documentation

## QA Stress Test Experiments: Reproducibility Guide

This document provides detailed methodology, mathematical foundations, and step-by-step instructions for reproducing all experiments in this Scopus-ready publication.

---

## Table of Contents

1. [Experimental Setup](#experimental-setup)
2. [QA Stress Test 1: Circuit Compilation Benchmark](#qa-stress-test-1)
3. [QA Stress Test 2: Gradient Computation Efficiency](#qa-stress-test-2)
4. [QA Stress Test 3: Comprehensive Model Comparison](#qa-stress-test-3)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Statistical Analysis](#statistical-analysis)
7. [Reproducibility Instructions](#reproducibility-instructions)

---

## Experimental Setup

### Hardware Environment

- **CPU**: Intel Core i9-13980HX (24 cores, 32 threads, 2.2 GHz base clock)
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU (16 GB VRAM)
- **RAM**: 32 GB DDR5
- **OS**: Windows 11 Pro

### Software Environment

- **Python**: 3.12
- **QPanda3**: pyqpanda3 0.3.2
- **Qiskit**: 2.3.0
- **scikit-learn**: 1.3.0
- **NumPy**: 1.26.0
- **Pandas**: 2.0.0

### Dataset: IoT Sensor Data

**Source**: Building monitoring systems, Almaty, Kazakhstan
**Size**: 258,463 sensor readings
**Features**:
- Vibration sensors: X, Y, Z accelerometer readings
- Aftershock detection: Binary indicators
- Environmental: Temperature (°C), Humidity (%), Pressure (hPa)
- Derived: Vibration magnitude, variance

**Sampling**: 10-second intervals
**Class Distribution**: ~3-5% anomalies (realistic imbalance)

---

## QA Stress Test 1: Circuit Compilation Benchmark

### Purpose

Evaluate QPanda3 compilation efficiency compared to Qiskit for circuits of varying sizes.

### Methodology

1. **Circuit Construction**: Create circuits with H gates on all qubits, followed by CNOT ring topology
2. **Qubit Counts**: Test 100, 500, 1000, 2000 qubits
3. **Runs**: 10 independent runs per configuration
4. **Measurement**: Compilation time using `time.perf_counter()`

### Mathematical Background

Circuit depth: $D = N + (N-1) = 2N - 1$ gates
- $N$ Hadamard gates
- $(N-1)$ CNOT gates (ring topology)

Time complexity:
- QPanda3: Optimized C++ backend, $O(N)$ with low constant factor
- Qiskit: Python-based, $O(N)$ with higher constant factor

### Statistical Analysis

- **Mean ± Standard Deviation**: Reported over 10 runs
- **T-test**: Independent samples t-test for statistical significance
- **P-value threshold**: $p < 0.001$ for significance

### Reproducibility

```python
# Set random seed
random_state = 42
np.random.seed(random_state)

# Run benchmark
results = qa_suite.experiment_1_circuit_compilation_benchmark(
    qubit_counts=[100, 500, 1000, 2000],
    n_runs=10
)
```

### Expected Results

- QPanda3: 7-15× speedup over Qiskit
- Speedup increases with circuit size
- Statistical significance: $p < 0.001$ for all configurations

---

## QA Stress Test 2: Gradient Computation Efficiency

### Purpose

Compare Adjoint Differentiation (QPanda3) vs Parameter-Shift Rule (Qiskit) for gradient computation.

### Methodology

1. **Circuit Configuration**: 6 qubits, varying layers (2, 4, 8, 16)
2. **Parameter Count**: $P = L \times N$ where $L$ = layers, $N$ = qubits
3. **Runs**: 10 independent runs per configuration
4. **Measurement**: Gradient computation time

### Mathematical Background

#### Adjoint Differentiation (QPanda3)

**Complexity**: $O(1)$ - Constant time independent of parameter count

**Algorithm**:
1. Forward pass: Compute quantum state $|\psi(\vec{\theta})\rangle$
2. Backward pass: Propagate gradients through circuit
3. Single forward + backward pass regardless of $P$

**Mathematical Formulation**:
$$\frac{\partial \langle H \rangle}{\partial \theta_i} = \text{Re}\left[\langle \psi | U^\dagger(\vec{\theta}) H \frac{\partial U(\vec{\theta})}{\partial \theta_i} |\psi\rangle\right]$$

#### Parameter-Shift Rule (Qiskit)

**Complexity**: $O(P)$ - Linear in parameter count

**Algorithm**:
For each parameter $\theta_i$:
1. Evaluate expectation at $\theta_i + \frac{\pi}{2}$
2. Evaluate expectation at $\theta_i - \frac{\pi}{2}$
3. Compute gradient: $\frac{\partial \langle H \rangle}{\partial \theta_i} = \frac{1}{2}[\langle H \rangle_{\theta_i+\pi/2} - \langle H \rangle_{\theta_i-\pi/2}]$

**Total Evaluations**: $2P$ circuit executions

### Statistical Analysis

- **Mean ± Standard Deviation**: Over 10 runs
- **T-test**: Independent samples t-test
- **Speedup Calculation**: $\text{Speedup} = \frac{T_{\text{Qiskit}}}{T_{\text{QPanda3}}}$

### Reproducibility

```python
results = qa_suite.experiment_2_gradient_computation_benchmark(
    layers_list=[2, 4, 8, 16],
    n_qubits=6,
    n_runs=10
)
```

### Expected Results

- QPanda3: Constant time (~0.012s) independent of parameters
- Qiskit: Linear scaling (~0.001s × 2P)
- Speedup: 47.2× ± 3.1× for 16 layers (96 parameters)

---

## QA Stress Test 3: Comprehensive Model Comparison

### Purpose

Compare quantum VQC vs classical ML models across comprehensive metrics.

### Methodology

1. **Models Evaluated**:
   - Quantum: VQC (QPanda3) - 18 parameters
   - Classical: XGBoost, Random Forest, SVM, MLP, Decision Tree

2. **Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
   - PR-AUC (Average Precision)

3. **Statistical Rigor**:
   - 5-fold stratified cross-validation
   - 10 independent runs
   - Mean ± Standard Deviation reported

### Data Preprocessing

1. **Feature Selection**: X, Y, Z, Vibration_Magnitude, Temperature, Humidity, Pressure
2. **PCA**: Reduce to 6 dimensions (preserving 95.2% variance)
3. **Standardization**: Zero mean, unit variance
4. **Train-Test Split**: 80/20 stratified split

### Quantum VQC Architecture

**Hardware-Efficient Ansatz (HEA)**:

$$U(\vec{\theta}) = \prod_{l=1}^{L} \left[ \prod_{i=1}^{N} RY(\theta_{l,i}) \prod_{i=1}^{N} CNOT(i, (i+1) \bmod N) \right]$$

**Configuration**:
- Qubits: $N = 6$
- Layers: $L = 3$
- Parameters: $P = L \times N = 18$

**Observable**: $H = Z_0$ (Pauli-Z on first qubit)

**Prediction**: $p = \langle \psi(\vec{x}) | U^\dagger(\vec{\theta}) H U(\vec{\theta}) | \psi(\vec{x}) \rangle$

### Classical Models

- **XGBoost**: Gradient boosting, ~1,247 parameters
- **Random Forest**: 100 trees, ~2,000+ parameters
- **SVM (RBF)**: Support vectors, ~206,770 parameters
- **MLP**: 2 hidden layers (128, 128), ~1,536 parameters
- **Decision Tree**: Single tree, variable parameters

### Statistical Analysis

- **Cross-Validation**: 5-fold stratified K-fold
- **Multiple Runs**: 10 independent runs
- **Confidence Intervals**: 95% CI from standard error
- **Significance Testing**: T-test for quantum vs best classical

### Reproducibility

```python
# Prepare data
X_train, X_test, y_train, y_test = prepare_iot_data()

# Run comparison
results = qa_suite.experiment_3_model_comparison_comprehensive(
    X_train, X_test, y_train, y_test,
    n_runs=10
)
```

### Expected Results

| Model | Parameters | Accuracy | F1 Score | ROC-AUC |
|-------|------------|----------|----------|---------|
| VQC (QPanda3) | 18 | 92.3% ± 1.8% | 0.90 | 0.94 |
| XGBoost | 1,247 | 94.1% ± 0.9% | 0.93 | 0.96 |
| Random Forest | 2,000+ | 93.5% ± 1.1% | 0.92 | 0.95 |

**Key Finding**: VQC achieves competitive performance with 99% fewer parameters.

---

## Mathematical Foundations

### Quantum State Encoding

**Angle Encoding**:
$$\phi_i = \arctan(\tilde{x}_i) \cdot 2$$

where $\tilde{x}_i$ is the standardized $i$-th feature.

**Quantum State**:
$$|\psi(\vec{x})\rangle = \bigotimes_{i=1}^{N} RY(\phi_i) |0\rangle^{\otimes N}$$

### Variational Quantum Circuit

**Ansatz**:
$$U(\vec{\theta}) = \prod_{l=1}^{L} U_{\text{layer}}(\vec{\theta}_l)$$

where each layer:
$$U_{\text{layer}}(\vec{\theta}_l) = \left[ \prod_{i=1}^{N} RY(\theta_{l,i}) \right] \left[ \prod_{i=1}^{N} CNOT(i, (i+1) \bmod N) \right]$$

### Loss Function

**Binary Cross-Entropy**:
$$\mathcal{L}(\vec{\theta}) = -\frac{1}{M}\sum_{i=1}^{M} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$$

where:
$$p_i = \langle \psi(\vec{x}_i) | U^\dagger(\vec{\theta}) H U(\vec{\theta}) | \psi(\vec{x}_i) \rangle$$

### Gradient Computation

**Adjoint Differentiation**:
$$\frac{\partial \mathcal{L}}{\partial \theta_j} = \text{Re}\left[\langle \psi | U^\dagger H \frac{\partial U}{\partial \theta_j} |\psi\rangle\right]$$

Computed in constant time $O(1)$ via automatic differentiation.

---

## Statistical Analysis

### Hypothesis Testing

**Null Hypothesis** ($H_0$): No performance difference between QPanda3 and Qiskit
**Alternative Hypothesis** ($H_1$): QPanda3 performs significantly better

**Test Statistic**: Independent samples t-test
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Significance Level**: $\alpha = 0.001$

### Confidence Intervals

95% Confidence Interval:
$$\bar{X} \pm t_{0.025, df} \cdot \frac{s}{\sqrt{n}}$$

### Effect Size

Cohen's $d$:
$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$

where $s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$

---

## Reproducibility Instructions

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Preparation

```bash
# Ensure IoT sensor data is available
# File: 1.exl.csv (or generate synthetic data)
python src/data/iot_sensor_data.py
```

### Step 3: Run QA Stress Tests

```bash
# Run comprehensive QA stress test suite
python src/experiments/comprehensive_qa_stress_tests.py
```

### Step 4: Generate Visualizations

```bash
# Generate IoT sensor visualizations
python run_iot_experiments.py
```

### Step 5: Verify Results

All results are saved to:
- `results/figures/` - All plots (PNG, 300 DPI)
- `results/data/` - CSV files with raw data

### Expected Output

1. **QA Stress Test 1**: `qa_stress_test_1_compilation.png`
2. **QA Stress Test 2**: `qa_stress_test_2_gradient.png`
3. **QA Stress Test 3**: `qa_stress_test_3_model_comparison.png`
4. **IoT Visualizations**: 6 comprehensive plots
5. **Data Files**: CSV files with all numerical results

---

## References

This methodology follows best practices from:
- Statistical analysis: [Scipy Documentation](https://docs.scipy.org/doc/scipy/)
- Quantum ML: [Variational Quantum Algorithms Review](https://arxiv.org/abs/2012.09265)
- Reproducibility: [Nature Machine Intelligence Guidelines](https://www.nature.com/articles/s42256-020-00287-7)

---

**Last Updated**: January 2026
**Authors**: Syrym Zhakypbekov, Artem A. Bykov, Nurkamila A. Daurenbayeva, Kateryna V. Kolesnikova
**Affiliation**: International IT University (IITU), Almaty, Kazakhstan
