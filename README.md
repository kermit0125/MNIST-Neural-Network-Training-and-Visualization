# 🧠 MNIST Neural Network Training and Visualization

A Python project demonstrating training of neural networks on MNIST digit data using **NumPy** and **PyTorch**, with visualizations including:

- PCA projection of MNIST test data
- Decision boundary between digits
- Loss curves for both frameworks
- Automated PDF report generation

---

## 🚀 Features

1. **NumPy Neural Network Training**  
   - Implements a simple two-layer neural network from scratch
   - Trains on MNIST data
   - Logs loss and accuracy per epoch

2. **PyTorch Neural Network Training**  
   - Uses PyTorch to implement a simple MLP
   - Trains on MNIST data
   - Logs loss and accuracy per epoch

3. **PCA Plot**  
   - Reduces MNIST test data to 2D via PCA
   - Visualizes digit clusters in 2D space

4. **Decision Boundary Plot**  
    - Trains a logistic regression classifier in PCA space
    - Visualizes the decision boundary between digits 3 and 8

5. **PDF Report Generation**  
   - Combines plots, metrics, and results into a single PDF report

---

## 📂 Project Structure
```
project_root/
│
├── data/
│ mnist.npz
│
├── figures/
│ pca_plot.png
│ decision_boundary_3_vs_8.png
│
├── outputs/
│ numpy_nn_loss_curve.png
│ numpy_nn_accuracy.txt
│ pytorch_nn_loss_curve.png
│ pytorch_nn_accuracy.txt
│ report.pdf
│
├── src/
│ main.py
│ train_numpy.py
│ train_pytorch.py
│ model_numpy.py
│ model_pytorch.py
│ utils_numpy.py
│ utils_pytorch.py
│ pca_plot.py
│ decision_boundary.py
│ report_generator.py
│
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

### 1. Create a virtual environment

**Windows (PowerShell):**

```
python -m venv myenv
.\myenv\Scripts\activate
```

**macOS / Linux:**

```
python3 -m venv myenv
source myenv/bin/activate
```

---

### 2. Install requirements

```
pip install -r requirements.txt
```


---

## 📝 How to Run

Run the main pipeline:
```
python src/main.py
```

All results will be saved under the `outputs/` and `figures/` folders. A PDF report will be generated summarizing all results.

---

## 📊 Example Outputs

Files generated after running:

- figures/pca_plot.png  
- figures/decision_boundary_3_vs_8.png  
- outputs/numpy_nn_loss_curve.png  
- outputs/numpy_nn_accuracy.txt  
- outputs/pytorch_nn_loss_curve.png  
- outputs/pytorch_nn_accuracy.txt  
- outputs/report.pdf

---

## 💻 Requirements

- Python >= 3.8
- numpy
- matplotlib
- scikit-learn
- torch
- fpdf

---

## License

MIT License

---

## Author

Keming Xing
