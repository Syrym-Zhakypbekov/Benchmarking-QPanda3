import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyqpanda3.core import QCircuit as QP_Circuit, QProg as QP_Prog, H as QP_H, CNOT as QP_CNOT, RX as QP_RX, RY as QP_RY, CPUQVM
from pyqpanda3.vqcircuit import VQCircuit, DiffMethod
from pyqpanda3.hamiltonian import Hamiltonian
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

def benchmark_circuit_construction(qubit_counts=[100, 500, 1000, 2000]):
    print("\n--- 1. Stress Test: Circuit Construction ---")
    results = []
    
    for n_qubits in qubit_counts:
        print(f"Testing {n_qubits} qubits...")
        
        # --- QPanda3 ---
        start = time.time()
        circ_qp = QP_Circuit()
        for i in range(n_qubits):
            circ_qp << QP_H(i)
        for i in range(n_qubits - 1):
            circ_qp << QP_CNOT(i, i+1)
        qp_duration = time.time() - start
        results.append({"SDK": "QPanda3", "Qubits": n_qubits, "Time (s)": qp_duration})
        
        # --- Qiskit ---
        start = time.time()
        circ_qk = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            circ_qk.h(i)
        for i in range(n_qubits - 1):
            circ_qk.cx(i, i+1)
        qk_duration = time.time() - start
        results.append({"SDK": "Qiskit", "Qubits": n_qubits, "Time (s)": qk_duration})
    
    # Add dummy results if one fails or is too fast to measure? No, just plot.
    df = pd.DataFrame(results)
    
    # Plot
    plt.figure()
    sns.lineplot(data=df, x="Qubits", y="Time (s)", hue="SDK", marker="o", linewidth=2.5)
    plt.title("Experimental Results: Large-Scale Circuit Construction Speed", fontsize=16, fontweight='bold')
    plt.ylabel("Time (seconds) - Lower is Better")
    plt.xlabel("Number of Qubits")
    plt.yscale('log')
    plt.savefig("benchmark_circuit_construction.png")
    print("Saved benchmark_circuit_construction.png")
    return df

def benchmark_gradient_calc(layers_list=[2, 4, 8, 16]):
    print("\n--- 2. Stress Test: VQC Gradient/Update Overhead ---")
    results = []
    
    n_qubits = 6
    
    for layers in layers_list:
        print(f"Testing {layers} layers...")
        
        # --- QPanda3 VQC ---
        # Measure TRUE gradient calculation time
        vqc = VQCircuit()
        vqc.set_Param([layers, n_qubits]) 
        for l in range(layers):
            for q in range(n_qubits):
                vqc << QP_RX(q, vqc.Param([l, q]))
            for q in range(n_qubits - 1):
                vqc << QP_CNOT(q, q+1)
        
        ham_dict = {f"Z{q}": 1.0 for q in range(n_qubits)}
        ham = Hamiltonian(ham_dict)
        params = list(np.random.random(layers * n_qubits))
        
        start = time.time()
        # Adjoint differentiation, usually very fast
        grad = vqc.get_gradients(params, ham, diff_method=DiffMethod.ADJOINT_DIFF)
        qp_duration = time.time() - start
        
        results.append({"SDK": "QPanda3", "Layers": layers, "Time (s)": qp_duration})
        
        # --- Qiskit Overhead ---
        # Measure Parameter Binding (AssignParameters) as a proxy for update cost
        start = time.time()
        qc = QuantumCircuit(n_qubits)
        qk_params = [Parameter(f"p_{i}") for i in range(layers * n_qubits)]
        
        idx = 0
        for l in range(layers):
            for q in range(n_qubits):
                qc.rx(qk_params[idx], q)
                idx += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q+1)
                
        # Simulate the binding overhead for 2*N parameter shifts (standard gradient calculation loop)
        # 1 binding per shift. (Parameter Shift Rule requires 2 evaluations per parameter)
        num_params = len(qk_params)
        for _ in range(num_params * 2):
             _ = qc.assign_parameters({p: np.random.random() for p in qk_params})
             
        qk_duration = time.time() - start
        results.append({"SDK": "Qiskit (Update Overhead)", "Layers": layers, "Time (s)": qk_duration})

    df = pd.DataFrame(results)
    
    # Plot
    plt.figure()
    sns.barplot(data=df, x="Layers", y="Time (s)", hue="SDK", palette="viridis")
    plt.title("VQC Optimization Loop Overhead (Gradient vs Binding)", fontsize=16, fontweight='bold')
    plt.ylabel("Time (seconds) - Lower is Better")
    plt.savefig("benchmark_gradient.png")
    print("Saved benchmark_gradient.png")
    return df

if __name__ == "__main__":
    print("Starting Heavy QA Stress Test...")
    try:
        benchmark_circuit_construction()
        benchmark_gradient_calc()
        print("\nStress Test Complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
