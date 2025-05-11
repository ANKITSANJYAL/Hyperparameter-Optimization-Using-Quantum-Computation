from itertools import product
import numpy as np
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from qiskit.quantum_info import Pauli, SparsePauliOp
# from qiskit.primitives import Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from src.model_runner import run_mlp
from time import time

def decode(bitstring):
    lr = 0.01 if bitstring[0] == 0 else 0.1
    layers = 1 if bitstring[1] == 0 else 2
    act = 'relu' if bitstring[2] == 0 else 'tanh'
    return (lr,layers,act)

def build_cost_operator(cost_dict):
    n = 3
    pauli_terms = []

    for bitstring , cost in cost_dict.items():
        z_string = ''.join(['I' if b==0 else 'Z' for b in bitstring[::-1]])
        pauli = Pauli(z_string)
        pauli_terms.append((cost,pauli))
    
    op = SparsePauliOp.from_list([(p.to_label(),c) for c,p in pauli_terms])

    return op

def quantum_search(X_train,y_train,X_val,y_val):
    configs = list(product([0,1],repeat=3))
    cost_dict = {}

    for bits in configs:
        lr,layers,act = decode(bits)
        acc = run_mlp(X_train,y_train,X_val,y_val,lr,layers,act)

        cost = 1 - acc
        cost_dict[bits] = cost

    op = build_cost_operator(cost_dict)
    sampler = AerSampler()
    qaoa = QAOA(optimizer=COBYLA(), sampler=sampler, reps=1)


    start_time = time()
    result = qaoa.compute_minimum_eigenvalue(operator=op)
    end_time = time()

    optimal_bitstring = np.round(result.optimal_point).astype(int)

    decoded_params = decode(optimal_bitstring)

    final_acc = run_mlp(X_train,y_train,X_val,y_val,*decoded_params)
    return decoded_params , final_acc , end_time - start_time