# src/report_generator.py

import os
from fpdf import FPDF


#Generate a PDF report of all results
def generate_report(output_dir, figures_dir):
    os.makedirs(output_dir, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "MNIST Neural Network Report", ln=True)

    # Add Numpy Loss Curve
    numpy_loss_path = os.path.join(output_dir, "numpy_nn_loss_curve.png")
    if os.path.exists(numpy_loss_path):
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Numpy NN Loss Curve:", ln=True)
        pdf.image(numpy_loss_path, w=150)
        pdf.ln(10)

    # Add PyTorch Loss Curve
    pytorch_loss_path = os.path.join(output_dir, "pytorch_nn_loss_curve.png")
    if os.path.exists(pytorch_loss_path):
        pdf.cell(0, 10, "PyTorch NN Loss Curve:", ln=True)
        pdf.image(pytorch_loss_path, w=150)
        pdf.ln(10)

    # Add PCA Plot
    pca_path = os.path.join(figures_dir, "pca_plot.png")
    if os.path.exists(pca_path):
        pdf.cell(0, 10, "PCA Plot:", ln=True)
        pdf.image(pca_path, w=150)
        pdf.ln(10)

    # Add Decision Boundary
    decision_path = os.path.join(figures_dir, "decision_boundary_3_vs_8.png")
    if os.path.exists(decision_path):
        pdf.cell(0, 10, "Decision Boundary (3 vs 8):", ln=True)
        pdf.image(decision_path, w=150)
        pdf.ln(10)

    # Add Accuracy Values
    for framework in ["numpy", "pytorch"]:
        acc_path = os.path.join(output_dir, f"{framework}_nn_accuracy.txt")
        if os.path.exists(acc_path):
            pdf.cell(0, 10, f"{framework.capitalize()} NN Accuracy per Epoch:", ln=True)
            with open(acc_path, "r") as f:
                for line in f.readlines():
                    pdf.cell(0, 10, line.strip(), ln=True)
            pdf.ln(10)

    report_path = os.path.join(output_dir, "report.pdf")
    pdf.output(report_path)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    # For standalone testing
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(base_dir, "figures")
    generate_report(output_dir=outputs_dir, figures_dir=figures_dir)
