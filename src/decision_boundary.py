# src/decision_boundary.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#Plotting the classification decision boundary in PCA space
def plot_decision_boundary(npz_path, digit1=3, digit2=8, save_path="figures/decision_boundary_3_vs_8.png"):
    # Load data
    data = np.load(npz_path)
    X = data["x_test"].reshape(-1, 28*28)
    y = data["y_test"]

    # Filter to 2 digits
    mask = np.logical_or(y == digit1, y == digit2)
    X = X[mask]
    y = y[mask]
    y = (y == digit2).astype(int)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Train a simple classifier (Fitting Boundaries)
    clf = LogisticRegression()
    clf.fit(X_2d, y)

    # Generate mesh grid
    x_min, x_max = X_2d[:, 0].min()-1, X_2d[:, 0].max()+1
    y_min, y_max = X_2d[:, 1].min()-1, X_2d[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    #Draw the regional distribution (we choose 6 and 8 here)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap=plt.cm.coolwarm, s=10, edgecolors='k')
    plt.title(f"Decision Boundary between {digit1} and {digit2}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Decision boundary saved to {save_path}")
