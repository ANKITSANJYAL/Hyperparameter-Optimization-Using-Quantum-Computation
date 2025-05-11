from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_binary_mnist():
    print("Loading Mnist ..")
    X , y = fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)

    mask = np.isin(y,['0','1'])
    X,y = X[mask] , y[mask].astype(int)


    #Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Train/Val Split
    X_train,X_val,y_train,y_val = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

    return X_train,X_val,y_train,y_val

