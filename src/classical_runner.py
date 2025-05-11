from itertools import product
from time import time
from src.model_runner import run_mlp

def classical_grid_search(X_train,y_train,X_val,y_val):
    search_space = {
        'lr' : [0.01,0.1],
        'layers' : [1,2],
        'activation' : ['relu','tanh']
    }
    
    best_acc = 0
    best_config = None
    results = []


    start_time = time()
    
    for lr,layers,activation in product(search_space['lr'],search_space['layers'],search_space['activation']):
        acc = run_mlp(X_train,y_train,X_val,y_val,lr,layers,activation)
        results.append((lr,layers,activation,acc))
        if acc>best_acc:
            best_acc = acc
            best_config = (lr,layers,activation)
    total_time = time() - start_time
    return best_config , best_acc , total_time , results