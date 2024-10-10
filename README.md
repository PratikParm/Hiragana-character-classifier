# Hiragana-character-classifier
A CNN to classify Japanese Hiragana characters

# KMNIST Classification using LeNet

This project implements a simple deep learning model (LeNet) to classify Japanese Hiragana characters. The model is built using PyTorch and is trained on the KMNIST dataset. It provides a training pipeline, model evaluation, and visualisation of training history.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/PratikParm/Hiragana-character-classifier.git
    cd Hiragana-character-classifier
    ```

2. **Set up a Python environment**:
    Create a virtual environment and install the required packages:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Install PyTorch**:
    Depending on your system's CUDA capabilities, follow [PyTorch's installation guide](https://pytorch.org/get-started/locally/) to install the correct version for your setup.

## Project Structure
```
├── data/                   # KMNIST dataset directory
├── output/                 # Directory where the model and plots are saved
├── pyimagesearch/           # Contains the implementation of LeNet model
│   └── lenet.py
├── train.py                # Script to train the model
├── README.md               # This file
├── requirements.txt        # Dependencies file
```


## Usage
To train the model and generate a plot of loss and accuracy, use the following command:

```bash
python train.py --model output/lenet_kmnist.pth --plot output/lenet_kmnist_plot.png
```

### Arguments
- `--model`: Path to save the trained model.
- `--plot`: Path to save the training loss/accuracy plot.

### Hyperparameters
- Initial Learning Rate: `1e-3`
- Batch Size: `64`
- Epochs: `10`
- Train/Val Split: `75%/25%`

You can adjust these hyperparameters in the train.py script.

## Results
The training process generates a plot for training and validation loss/accuracy, as well as a classification report for the model’s performance on the test dataset. These results help visualize model performance and determine areas for improvement.

### Sample Classification Report:

```
              precision    recall  f1-score   support

           o       0.11      0.11      0.11      1000
          ki       0.09      0.09      0.09      1000
          su       0.10      0.10      0.10      1000
         tsu       0.10      0.10      0.10      1000
          na       0.10      0.10      0.10      1000
          ha       0.11      0.10      0.10      1000
          ma       0.10      0.10      0.10      1000
          ya       0.11      0.11      0.11      1000
          re       0.10      0.10      0.10      1000
          wo       0.10      0.10      0.10      1000

    accuracy                           0.10     10000
   macro avg       0.10      0.10      0.10     10000
weighted avg       0.10      0.10      0.10     10000
```
### Sample plot: 

![plot](https://github.com/user-attachments/assets/56454ac2-0339-444a-9e24-5add08d2ac54)


## License
This project is licensed under the MIT License. See the LICENSE file for details.
