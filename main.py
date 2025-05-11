from src.data_loader import load_binary_mnist
# from src.model_runner import run_mlp
from src.classical_runner import classical_grid_search
from src.quantum_optimizer import quantum_search


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_binary_mnist()
    
    print("\n🔁 Classical Grid Search")
    best_classical, acc_classical, time_classical, _ = classical_grid_search(X_train, y_train, X_val, y_val)
    print(f"Best Classical Config: {best_classical}, Acc: {acc_classical:.4f}, Time: {time_classical:.2f}s")

    print("\n⚛️ Quantum QAOA Search")
    best_quantum, acc_quantum, time_quantum = quantum_search(X_train, y_train, X_val, y_val)
    print(f"Best Quantum Config: {best_quantum}, Acc: {acc_quantum:.4f}, Time: {time_quantum:.2f}s")
