# Neural Network Classifiers for EMNIST

This repository contains a collection of experiments that build, tune, and evaluate neural network classifiers using PyTorch on the [EMNIST Balanced dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset) (Extended MNIST for handwritten characters). Both Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs) are explored, with extensive hyperparameter tuning and performance evaluation using k-fold cross-validation.

---

## Overview

In this project, we implement two primary approaches:

- **CNN Classifier:**  
  A baseline CNN model is built with two convolutional layers (with batch normalization), max pooling, and two fully connected layers. We experiment with different optimizers (SGD, RMSprop, Adam), activation functions (ReLU, ELU, LeakyReLU), learning rate schedulers (StepLR and MultiStepLR), and the use (or not) of batch normalization. Finally, the best hyperparameters are incorporated into a final CNN model whose performance is evaluated on the test set.

- **MLP Classifier:**  
  A simple feed-forward neural network (MLP) is implemented to classify the flattened images from the EMNIST dataset. The model architecture is varied in terms of the number of hidden layers, hidden neurons, activation function, dropout rate, optimizer choice, and learning rate. Grid-search with k-fold cross-validation is used to find the best configuration. The final tuned MLP is then evaluated on the test data.

Both models are evaluated using standard metrics such as accuracy, precision, recall, F1 score, and confusion matrices. Visualizations (loss vs. epoch, accuracy vs. epoch, and sample prediction plots) are generated to illustrate model performance.

---

### Data Preparation

- **EMNIST Dataset:**  
  The project uses the EMNIST *balanced* split. The dataset is automatically downloaded into the `./data` folder when you run the training scripts.  
- **Mapping File:**  
  Ensure that the `emnist-balanced-mapping.txt` file (which maps label indices to characters) is placed inside the `data/` folder.

### Training and Hyperparameter Tuning

Both the CNN and MLP experiments use k-fold cross-validation (with *k* = 3) to explore various hyperparameters:
  
- **CNN Experiments:**  
  - **Baseline CNN:** A straightforward CNN architecture is implemented.
  - **Optimizer Exploration:** Testing different optimizers (Adam, SGD, RMSprop).
  - **Activation Function Experiments:** Comparing ReLU, ELU, and LeakyReLU.
  - **Learning Rate Scheduler & Batch Normalization:** Grid search over different schedulers (StepLR, MultiStepLR) and with/without batch normalization.
  - **Final CNN Model:** Best hyperparameters (learning rate, optimizer, weight decay, dropout, etc.) are incorporated into the final CNN architecture.  
  Run the CNN training script:
  ```bash
  python src/train_cnn.py
  ```

- **MLP Experiments:**  
  - **Baseline MLP:** A feed-forward network with variable hidden layers and neurons.
  - **Hyperparameter Exploration:** Grid search over number of hidden layers, hidden neurons, activation function (ReLU, ELU, LeakyReLU), dropout rate, learning rate, and optimizer choice.
  - **Final MLP Model:** The best performing hyperparameters are used for the final model, and performance is evaluated on the test set.  
  Run the MLP training script:
  ```bash
  python src/train_mlp.py
  ```

### Evaluation

- **Metrics:**  
  Models are evaluated in terms of loss, accuracy, precision, recall, and F1 score. Confusion matrices are generated for a visual comparison of predicted versus true labels.
- **Visualization:**  
  Training progress is visualized with loss vs. epoch and accuracy vs. epoch plots. Additionally, sample predictions (with mapped character labels) are plotted along with the true labels.
- **Notebook Export:**  
  To convert the Jupyter Notebook version of the project to PDF, run:
  ```bash
  jupyter nbconvert --to pdf <notebook_name>.ipynb
  ```

---

## Results

### CNN Model

- **Final Performance on Test Set:**  
  - Test Loss: ~0.5116  
  - Test Accuracy: ~94.92%  
- **Evaluation Metrics:**  
  - Precision: ~0.9139  
  - Recall: ~0.9062  
  - F1 Score: ~0.9053  
- **Observations:**  
  Hyperparameter tuning with grid search and k-fold cross-validation led to notable improvements in accuracy and robustness, with the final CNN model outperforming several experimental variants.

### MLP Model

- **Final Performance on Test Set:**  
  - Test Loss: ~0.3417  
  - Test Accuracy: ~78.85%  
- **Evaluation Metrics:**  
  - Precision, Recall, and F1 metrics are calculated and reported in the final summary.
- **Observations:**  
  Although the MLP model did not reach the accuracy levels of the CNN classifier, the grid search experimentation provided insights into optimal configurations (e.g., using ELU activation, 4 hidden layers, 128 hidden neurons, a dropout of 0.2, and SGD with a learning rate of 0.1).

Detailed loss and accuracy plots for both models are generated and stored in the `outputs/loss_plots` directory.

---

## Experiments & Hyperparameter Tuning

The project includes extensive experiments to determine the best model configurations. Experiments include:

- **CNN:**
  - Tuning learning rates, dropout rates, and weight decay.
  - Comparing optimizers (Adam, SGD, RMSProp) using k-fold CV.
  - Exploring different activation functions.
  - Evaluating the impact of learning rate schedulers and the presence of batch normalization.

- **MLP:**
  - Varying the number of hidden layers and hidden neurons.
  - Evaluating different activation functions (ReLU, ELU, LeakyReLU) and dropout strategies.
  - Testing different optimizers and learning rates.

These experiments are implemented using nested loops over hyperparameter grids, with performance tracked for each configuration so that the final model represents the best performing setup.

---

## Contributing

Contributions are welcome! Please submit pull requests or open issues if you have suggestions, bug reports, or improvements for this project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/clareokafor/Neural-Network-for-Classification/blob/main/LICENSE) file for more details.

---

## Acknowledgements

- [EMNIST Dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
- PyTorch and Torchvision teams for providing powerful deep learning tools.
- The scikit-learn community for easy-to-use model selection and evaluation utilities.

