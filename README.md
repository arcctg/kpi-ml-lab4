# Scikit-Learn Handwritten Digit Recognition: A Multi-Classifier Benchmark

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arcctg/kpi-ml-lab4/blob/main/01_digits_classifier_benchmark.ipynb)

## Project Overview

This project performs a comprehensive benchmark of multiple machine learning classification algorithms on the canonical MNIST-style **Digits dataset** from scikit-learn. Starting from a simple K-Nearest Neighbors baseline, the analysis progresses through rigorous hyperparameter tuning with `GridSearchCV`, a detailed comparison against a Support Vector Classifier (SVC) and Gaussian Naive Bayes, and culminates in a deep diagnostic investigation of each model's errors using confusion matrices, classification reports, and visual inspection of misclassified images.

The ultimate goal is not only to find the best-performing classifier, but to understand *why* certain models outperform others and how the choice of hyperparameters affects robustness and generalization.

## Key Features

* **Dataset Visualization:** Loading scikit-learn's built-in Digits dataset (1,797 samples of 8×8 handwritten digit images) and visualizing samples in a grid to understand the underlying data.
* **Baseline KNN Classification:** Training a default `KNeighborsClassifier` (k=5), evaluating its test accuracy (~98.6%), and identifying individual misclassified samples with their indices.
* **Confusion Matrix Analysis:** A custom heatmap function (`plot_confusion_matrix`) renders confusion matrices for visual, per-class error analysis, revealing which digit pairs are most frequently confused (e.g., 3↔5, 9↔8).
* **Classification Report Deep Dive:** Detailed per-class precision, recall, and F1-score analysis, explaining the trade-offs between models (e.g., high-precision-low-recall vs. high-recall-low-precision behavior for specific digits).
* **Hyperparameter Tuning (GridSearchCV):** Exhaustive 5-fold cross-validation grid search over `n_neighbors`, `weights`, and `p` for KNN, and `C`, `kernel`, and `gamma` for SVC.
* **Multi-Model Comparison:** Side-by-side comparison of Optimized KNN, Optimized SVC, and Baseline GaussianNB using both test set accuracy and rigorous 20-fold cross-validation scores.
* **KNN K-Value Impact Analysis:** Bar charts showing how accuracy changes for k values from 1 to 20 for both baseline (uniform) and optimized (distance-weighted) KNN, visually proving why distance weighting dramatically improves robustness.
* **Error Analysis with Image Visualization:** Visually inspecting the specific misclassified digit images to demonstrate that errors arise from inherently ambiguous, heavily pixelated, or non-standardly drawn digits.

## Technologies Used

* **Language:** Python
* **Machine Learning:** Scikit-Learn (`KNeighborsClassifier`, `SVC`, `GaussianNB`, `GridSearchCV`, `cross_val_score`)
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

## Dataset

* **Scikit-Learn Digits Dataset:** A built-in dataset containing 1,797 samples of handwritten digits (0–9). Each sample is an 8×8 grayscale image (64 features representing pixel intensities). The dataset is loaded via `sklearn.datasets.load_digits`.
* **Train/Test Split:** 80% training (1,437 samples) and 20% testing (360 samples), using `random_state=11` for reproducibility.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arcctg/digits-classifier-benchmark.git
   cd digits-classifier-benchmark
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Open the `01_digits_classifier_benchmark.ipynb` file in your browser to run the analysis cells.*

## Results and Insights

* **Baseline Performance:** Both KNN (k=5, uniform weights) and default SVC achieved an identical cross-validation accuracy of **~98.61%**. GaussianNB, acting as a simple probabilistic baseline, performed significantly lower at **~85.56%**.
* **Impact of Hyperparameter Tuning:**
  * **Optimized KNN** (k=3, distance weights, Euclidean metric) improved from 98.61% to **98.89%** cross-validation accuracy. Crucially, switching to distance-weighted voting made the model dramatically more *robust* — its accuracy remains stable even as `k` is increased to 16, unlike the uniform-weight baseline which degrades significantly.
  * **Optimized SVC** (C=1, gamma=0.001, RBF kernel) improved from 98.61% to **98.66%** in the final 20-fold cross-validation evaluation.
* **Final Verdict (20-Fold Cross-Validation):**
  * Optimized SVC: **98.66%** ← Winner
  * Optimized KNN: **98.38%**
  * The SVC is the superior production choice — it achieves the highest accuracy and has significantly faster inference time than KNN's lazy-learning distance computations.
* **Error Analysis:** Misclassifications are not random. Both models primarily confuse digits that share similar visual characteristics — rounded loops (3, 5, 8, 9) or sharp angular strokes (4 and 7). Visual inspection of misclassified images confirms that these are inherently ambiguous or poorly drawn samples that even a human observer might struggle to classify.

## License

This project is open-source and available under the [MIT License](LICENSE).
