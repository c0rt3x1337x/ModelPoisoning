# Model Poisoning in Multi-Class Algorithms

## Overview

This project investigates the concept of *Model Poisoning* in multi-class classification algorithms. Model poisoning refers to the intentional corruption of training data to degrade the performance of machine learning models. In multi-class classification, such attacks can introduce vulnerabilities, biases, or critical failures in systems designed to categorize inputs into multiple distinct classes.

The foundation of this project is a Convolutional Neural Network (CNN) implemented using PyTorch, trained on the MNIST dataset — a classic multi-class classification problem involving handwritten digit recognition (0–9). Building this baseline enables future work on systematically simulating poisoning attacks and studying their impact on model integrity.

## Literature Review

Following an in-depth review of key academic literature on data poisoning and model corruption techniques, we constructed this project to bridge theoretical knowledge with practical experimentation. 

Our references include seminal surveys and targeted research on poisoning attacks: [References](References.md).

### Relation to Implementation

Based on these references, we:
- Built a **clean CNN model** on the MNIST dataset using **PyTorch**, aligned with standard baselines used in prior poisoning research.
- Adopted training procedures, evaluation metrics, and model validation approaches similar to those outlined in recent academic benchmarks.
- Prepared the environment for **future poisoning experiments**, such as label flipping, pixel perturbation, and data injection.

Thus, our implementation represents a first, crucial step in a research-based exploration of adversarial poisoning attacks in deep learning.

## Objective

The primary objectives of this project are:

- **Implement** a clean multi-class classification model using CNNs.
- **Establish** a performance baseline on unpoisoned MNIST data.
- **Prepare** the model and dataset for later simulation of poisoning attacks.
- **Analyze** the effects of data poisoning on model performance.
- **Explore** potential mitigation and defense strategies.

## Key Concepts

- **Multi-Class Classification:** Machine learning task where each input is assigned one label out of multiple possible categories.
- **Model Poisoning Attack:** Adversarial technique where training data is manipulated to degrade model performance or introduce targeted vulnerabilities.
- **Data Integrity:** Assurance that the training data remains authentic and free from malicious tampering.

## Technologies Used

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib (for visualization)
- Jupyter Notebooks (for experimentation and documentation)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/model-poisoning-multiclass.git
    

2. Navigate to the project directory:
    ```bash
    cd model-poisoning-multiclass
    

3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies:
    ```bash
    pip install -r requirements.txt    


## Usage

1. Open Jupyter Notebook 
   ```bash
    jupyter notebook MNIST_CNN_Baseline.ipynb


