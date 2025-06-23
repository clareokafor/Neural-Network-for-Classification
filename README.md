
# Neural-Network-for-Classification Problem
## Handwritten Character Recognition: MLP vs CNN

This project explores and compares the performance of **Multi-Layer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)** for handwritten character recognition using the **EMNIST dataset**.

## 🧠 Project Overview

The core objective of this project is to evaluate the performance of MLP and CNN models on handwritten character recognition and identify which architecture offers superior accuracy and robustness. Using the EMNIST dataset, we implemented and trained both models under various hyperparameter configurations.

### Key Findings:

* **CNNs outperformed MLPs**, achieving a higher test accuracy of **87.88%** vs **83.65%** for MLPs.
* **CNNs effectively captured spatial features** in image data, leading to more accurate character classification.
* **MLPs** showed limitations in processing spatial features, resulting in slightly lower performance.
* **CNNs are more computationally intensive**, making them less suitable for resource-constrained environments.

## 📊 Dataset

We used the **[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)** (Extended MNIST) dataset, which contains balanced handwritten characters (letters and digits).

* Classes: 47 balanced character classes
* Format: Grayscale, 28x28 pixel images

## 🏗️ Models

### Multi-Layer Perceptron (MLP)

* Fully connected layers
* Requires flattening image data
* Struggles with spatial information

### Convolutional Neural Network (CNN)

* Convolutional + pooling layers
* Preserves spatial structure of images
* Robust and highly accurate for visual tasks

## ⚙️ Implementation

The main implementation is contained in `neuralnetworks_(1).py`. It includes:

* Data loading and preprocessing (normalization, reshaping)
* MLP and CNN architecture definitions
* Model training and evaluation
* Accuracy and loss visualization

## 🚀 Results

| Model | Test Accuracy |
| ----- | ------------- |
| MLP   | 83.65%        |
| CNN   | 87.88%        |

CNNs demonstrated **superior generalization** and **resilience to noise**, particularly useful in real-world applications.

## 🛠️ Requirements
Tested with:

* Python 3.8+
* TensorFlow 2.x
* NumPy, Matplotlib
* Python 3.8+
* Jupyter Notebook or JupyterLab

## 📂 File Structure

```
├── neuralnetworks.ipynb     # Main code file for MLP vs CNN comparison
├── README.md                  # Project documentation
```

## 📌 Conclusion

While both MLPs and CNNs can perform character recognition, **CNNs** are better suited for image-based tasks due to their ability to extract spatial features. However, **MLPs** still hold value in simpler or tabular tasks and resource-constrained environments.

## 📄 License

This project is open-source under the [MIT License](LICENSE).
