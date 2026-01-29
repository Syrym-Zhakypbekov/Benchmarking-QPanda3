# 🚀 Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/Syrym-Zhakypbekov/Benchmarking-QPanda3.git
cd Benchmarking-QPanda3

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
python src/utils/verify_qpanda.py
```

## Run Experiments

### Quick Benchmark
```bash
python main.py --experiment all
```

### Individual Experiments
```bash
# Circuit construction benchmark
python src/experiments/benchmark_stress_test.py

# VQC training
python src/experiments/run_vqc_experiment.py

# Comprehensive experiments
python src/experiments/run_comprehensive_experiments.py
```

## View Results

Results are saved to:
- `results/figures/` - All plots (PNG, 300 DPI)
- `results/data/` - CSV files with raw data

## Paper

The complete research paper is in `paper/` directory:
- `paper_ULTIMATE_scopus.pdf` - Final PDF version
- `paper_ULTIMATE_scopus.tex` - LaTeX source
- `paper_for_scopus_ULTIMATE.docx` - DOCX version

## Documentation

See `docs/` directory for:
- QA assessments
- Paper summaries
- Project documentation
