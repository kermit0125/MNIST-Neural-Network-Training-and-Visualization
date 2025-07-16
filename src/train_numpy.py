# src/train_numpy.py

import os
import numpy as np
import matplotlib.pyplot as plt

import utils_numpy
from model_numpy import NumpyNN

OUTCOME_DIR = "./outputs"

def train_numpy_nn(epochs=20, batch_size=128, hidden_size=128, learning_rate=0.01):
    os.makedirs(OUTCOME_DIR, exist_ok=True)

    # Data loading and data preprocessing
    x_train, y_train, x_test, y_test = utils_numpy.load_mnist()

    x_train = utils_numpy.preprocess_images(x_train)
    x_test = utils_numpy.preprocess_images(x_test)

    y_train_onehot = utils_numpy.one_hot_encode(y_train, num_classes=10)
    y_test_onehot = utils_numpy.one_hot_encode(y_test, num_classes=10)

    model = NumpyNN(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        learning_rate=learning_rate
    )

    train_losses = []
    test_accuracies = []


    #Training loop
    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)

        for i in range(0, x_train.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train_onehot[batch_idx]

            y_pred = model.forward(x_batch)
            loss = utils_numpy.cross_entropy_loss(y_pred, y_batch)
            model.backward(y_batch)

        # Evaluate
        y_test_pred = model.forward(x_test)
        test_loss = utils_numpy.cross_entropy_loss(y_test_pred, y_test_onehot)
        test_acc = utils_numpy.compute_accuracy(y_test_pred, y_test_onehot)

        train_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.2f}")

    # Save loss curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Test Loss", marker='o')
    plt.legend()
    plt.title("Numpy NN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, epochs+1))
    plt.grid(True)
    plt.savefig(os.path.join(OUTCOME_DIR, "numpy_nn_loss_curve.png"))
    plt.close()

    # Save accuracy
    with open(os.path.join(OUTCOME_DIR, "numpy_nn_accuracy.txt"), "w") as f:
        for acc in test_accuracies:
            f.write(f"{acc}\n")

    print("Training finished. Outputs saved in:", OUTCOME_DIR)

if __name__ == "__main__":
    train_numpy_nn()
