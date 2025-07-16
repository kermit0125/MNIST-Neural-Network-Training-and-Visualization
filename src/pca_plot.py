# src/pca_plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Plotting the PCA projection of the MNIST data
def plot_pca(npz_path, save_path):

    # Load data
    data = np.load(npz_path)
    X = data["x_test"].reshape(-1, 28*28)
    y = data["y_test"]

    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    #Draw a scatter plot
    plt.figure(figsize=(8,6))
    for digit in range(10):
        idxs = np.where(y == digit)
        plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], label=str(digit), s=10, alpha=0.6)
    plt.legend()
    plt.title("PCA Projection of MNIST Test Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved to {save_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    npz_path = os.path.abspath(os.path.join(base_dir, "../data/mnist.npz"))
    plot_pca(npz_path)
