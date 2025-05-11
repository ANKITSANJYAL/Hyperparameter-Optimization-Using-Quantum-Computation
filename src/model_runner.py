from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def run_mlp(X_train,y_train,X_val,y_val,lr=0.01,layers=1,activation='relu'):
    hidden_layers = tuple([64]*layers)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        learning_rate_init=lr,
        max_iter=300,
        random_state=42
    )
    clf.fit(X_train,y_train)
    preds = clf.predict(X_val)
    return accuracy_score(y_val,preds)