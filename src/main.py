# src/main.py

import os
import subprocess
from pca_plot import plot_pca
from decision_boundary import plot_decision_boundary
from report_generator import generate_report

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(base_dir, "figures")
    npz_path = os.path.join(base_dir, "data", "mnist.npz")

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    myenv_python = os.path.join(base_dir, "myenv", "Scripts", "python.exe")

    # 1. Train numpy model
    print("Running Numpy training...")
    subprocess.run([myenv_python, os.path.join(base_dir, "src", "train_numpy.py")])

    # 2. Train pytorch model
    print("Running PyTorch training...")
    subprocess.run([myenv_python, os.path.join(base_dir, "src", "train_pytorch.py")])

    # 3. PCA plot
    print("Generating PCA plot...")
    plot_pca(npz_path, save_path=os.path.join(figures_dir, "pca_plot.png"))

    # 4. Decision boundary
    print("Generating decision boundary plot...")
    plot_decision_boundary(
        npz_path,
        digit1=3,
        digit2=8,
        save_path=os.path.join(figures_dir, "decision_boundary_3_vs_8.png")
    )

    # 5. Generate PDF report
    print("Generating report...")
    generate_report(output_dir=outputs_dir, figures_dir=figures_dir)

if __name__ == "__main__":
    main()
