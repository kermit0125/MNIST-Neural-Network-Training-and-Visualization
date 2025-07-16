# /src/utils_numpy.py
import os
import numpy as np


#Initialize and provide NumPy version of utility functions
    #load_mnist             →   Read mnist.npz
    #preprocess_images      →   Standardize and flatten
    #one_hot_encode         →   One-hot encoding
    #cross_entropy_loss     →   Calculate cross entropy
    #compute_accuracy       →   Calculate accuracy
def load_mnist():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "../data/mnist.npz")
    data_path = os.path.abspath(data_path)

    data = np.load(data_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def preprocess_images(x):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(x.shape[0], -1)
    return x

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def compute_accuracy(y_pred, y_true):
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    acc = np.mean(pred_labels == true_labels)
    return acc

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    return loss
