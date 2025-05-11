# ⚛️ Quantum Optimizer for Hyperparameter Tuning

This project benchmarks **Quantum vs Classical hyperparameter tuning** for a binary image classification task using the MNIST dataset (digits 0 vs 1). The tuning explores:

- 🔁 Classical Grid Search
- ⚛️ Quantum Approximate Optimization Algorithm (QAOA)

---

## 🎯 Goal

Can quantum optimization algorithms find optimal hyperparameter configurations **more efficiently** than traditional methods?

We optimize:
- **Learning rate**: `0.01` or `0.1`
- **Number of layers**: `1` or `2`
- **Activation function**: `relu` or `tanh`

Total: `2 × 2 × 2 = 8` combinations

---

## 🛠️ Stack

| Component        | Tool Used                        |
|------------------|----------------------------------|
| Dataset          | MNIST (filtered: 0 vs 1)         |
| Classical Model  | `scikit-learn` (MLPClassifier)   |
| Quantum Backend  | `AerSampler` (Qiskit Aer)        |
| Quantum Optimizer| `QAOA` (`qiskit-algorithms`)     |
| Cost Modeling    | `SparsePauliOp` (manual QUBO)    |
| Environment      | Python 3.10 + Conda              |

---

## 🗂️ Project Structure

.
├── main.py                     # Entry point: runs both classical & quantum search
├── exploration.ipynb           # MNIST image + class balance visualization
├── src/
│   ├── data_loader.py          # Load + preprocess binary MNIST
│   ├── model_runner.py         # MLP model training + accuracy return
│   ├── classical_runner.py     # Grid search over all 8 combinations
│   └── quantum_optimizer.py    # QAOA using SparsePauliOp (Qiskit 2.0+ compatible)

## ✅ Current Status
 MNIST (0 vs 1) data loaded and explored

 Classical grid search implemented and benchmarked

 QAOA-based quantum optimizer implemented using SparsePauliOp

 Runtime and accuracy comparison working

🔮 Next Steps
 Log results to results.csv

 Visualize accuracy & runtime (bar chart)

 Add CLI / Streamlit dashboard

 Deploy on IBM QPU using qiskit-ibm-runtime
