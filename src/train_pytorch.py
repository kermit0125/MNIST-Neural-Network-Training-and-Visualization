# src/train_pytorch.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils_pytorch import get_dataloaders
from model_pytorch import MLP

# -------------------------------
# Training Function
# -------------------------------

def train_pytorch_nn():
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.01

    # Data loading and data preprocessing
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "../data/mnist.npz")
    data_path = os.path.abspath(data_path)

    output_dir = os.path.join(base_dir, "../outputs")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        npz_path=data_path
    )

    model = MLP()                                                   #Model
    criterion = nn.CrossEntropyLoss()                               #Criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)     #Optimizer

    epoch_losses = []
    epoch_accuracies = []


    #Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        test_accuracy = evaluate(model, test_loader)
        epoch_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {avg_loss:.4f} - Test Accuracy: {avg_loss:.2f}")

    # Save loss curve
    plot_path = os.path.join(output_dir, "pytorch_nn_loss_curve.png")
    plot_loss_curve(epoch_losses, num_epochs, plot_path)

    # Save accuracy values
    acc_path = os.path.join(output_dir, "pytorch_nn_accuracy.txt")
    save_accuracy(epoch_accuracies, acc_path)

    print(f"Loss curve saved to: {plot_path}")
    print(f"Accuracy values saved to: {acc_path}")

# -------------------------------
# Evaluation Function
# -------------------------------

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# -------------------------------
# Plot Loss Curve
# -------------------------------

def plot_loss_curve(losses, num_epochs, save_path):
    plt.figure()
    plt.plot(
        range(1, num_epochs+1),
        losses,
        label="Test Loss",
        marker='o'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PyTorch NN Loss Curve")
    plt.xticks(range(1, num_epochs+1))
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# -------------------------------
# Save Accuracy to TXT
# -------------------------------

def save_accuracy(acc_list, save_path):
    with open(save_path, "w") as f:
        for acc in acc_list:
            f.write(f"{acc/100:.4f}\n")

# -------------------------------
# Run Training
# -------------------------------

if __name__ == "__main__":
    train_pytorch_nn()