"""
Main entry point for Benchmarking QPanda3 project.

This script provides a command-line interface to run various experiments.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.verify_qpanda import verify_qpanda_installation


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmarking QPanda3: QA Stress Tests and Experiments"
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify QPanda3 installation'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['circuit', 'gradient', 'scaling', 'ansatz', 'hyperparameter', 'all'],
        default='all',
        help='Which experiment to run'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Verify installation if requested
    if args.verify:
        print("Verifying QPanda3 installation...")
        result = verify_qpanda_installation()
        if not result['success']:
            print(f"\n❌ Verification failed: {result['message']}")
            sys.exit(1)
        print("\n✅ Verification successful!")
        return
    
    # Run experiments
    print("="*60)
    print("Benchmarking QPanda3: QA Stress Tests")
    print("="*60)
    
    if args.experiment == 'circuit' or args.experiment == 'all':
        print("\nRunning Circuit Construction Benchmark...")
        from src.experiments.benchmark_stress_test import benchmark_circuit_construction
        benchmark_circuit_construction()
    
    if args.experiment == 'gradient' or args.experiment == 'all':
        print("\nRunning Gradient Computation Benchmark...")
        from src.experiments.benchmark_stress_test import benchmark_gradient_calc
        benchmark_gradient_calc()
    
    if args.experiment == 'scaling' or args.experiment == 'all':
        print("\nRunning Scaling Study...")
        # Import and run scaling study
        print("  (Scaling study script needs to be implemented)")
    
    print("\n✅ Experiments complete!")


if __name__ == "__main__":
    main()
