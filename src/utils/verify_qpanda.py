"""
Verify QPanda3 installation and basic functionality.

This script checks if QPanda3 is properly installed and can execute
basic quantum circuits.
"""

import sys
from typing import Dict, Any

try:
    from pyqpanda3.core import QCircuit, QProg, H, CNOT, measure, CPUQVM
    QPANDA_AVAILABLE = True
except ImportError:
    QPANDA_AVAILABLE = False


def verify_qpanda_installation() -> Dict[str, Any]:
    """
    Verify QPanda3 installation by running a GHZ state circuit.
    
    Returns:
        Dictionary with verification results:
            - success: bool
            - message: str
            - result_counts: dict (if successful)
    """
    if not QPANDA_AVAILABLE:
        return {
            'success': False,
            'message': 'QPanda3 module not found. Please install: pip install pyqpanda3',
            'result_counts': None
        }
    
    try:
        print("Verifying QPanda3 Installation...")
        
        # Create a quantum circuit
        circuit = QCircuit()
        
        # Construct GHZ state: |000⟩ + |111⟩
        circuit << H(0)         # Apply Hadamard gate on qubit 0
        circuit << CNOT(0, 1)   # Apply CNOT gate with control 0 and target 1
        circuit << CNOT(1, 2)   # Apply CNOT gate with control 1 and target 2
        
        # Create a quantum program and compose the circuit
        prog = QProg()
        prog << circuit
        
        # Add measure operations
        prog << measure(0, 0) << measure(1, 1) << measure(2, 2)
        
        # Create a QVM and execute
        qvm = CPUQVM()
        print("Executing Quantum Circuit (GHZ State)...")
        qvm.run(prog, 1000)
        result = qvm.result().get_counts()
        
        # Verify GHZ state (should be ~50% |000⟩ and ~50% |111⟩)
        print("\nCircuit Topology:")
        print(prog)
        print("\nMeasurement Results (1000 shots):")
        print(result)
        
        # Check if results are reasonable (GHZ state)
        total_000 = result.get('000', 0)
        total_111 = result.get('111', 0)
        total_shots = sum(result.values())
        
        if total_shots > 0:
            prob_000 = total_000 / total_shots
            prob_111 = total_111 / total_shots
            
            # GHZ state should have ~50% |000⟩ and ~50% |111⟩
            if prob_000 > 0.4 and prob_111 > 0.4:
                print("\n✅ SUCCESS: QPanda3 is working correctly!")
                print(f"   GHZ state verified: |000⟩={prob_000:.1%}, |111⟩={prob_111:.1%}")
                return {
                    'success': True,
                    'message': 'QPanda3 verification successful',
                    'result_counts': result
                }
            else:
                print("\n⚠️  WARNING: Results don't match expected GHZ state")
                return {
                    'success': False,
                    'message': 'QPanda3 installed but results unexpected',
                    'result_counts': result
                }
        else:
            print("\n❌ ERROR: No measurement results obtained")
            return {
                'success': False,
                'message': 'No measurement results',
                'result_counts': result
            }
            
    except Exception as e:
        print(f"\n❌ ERROR: An exception occurred during verification: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Exception: {str(e)}',
            'result_counts': None
        }


if __name__ == "__main__":
    result = verify_qpanda_installation()
    sys.exit(0 if result['success'] else 1)
