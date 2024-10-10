import matplotlib
matplotlib.use("Agg")

import os
import logging
from tqdm import tqdm
from datetime import datetime
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# Setting up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
logger.info(f"Hyperparameters: \n"
            f"Initial Learning rate = {INIT_LR} \n"
            f"Batch size = {BATCH_SIZE} \n"
            f"Number of epochs = {EPOCHS} \n")

# Define tran-val split
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
logger.info(f'Train-val split = {TRAIN_SPLIT:.2f}-{VAL_SPLIT:.2f}')

# Set the device we will use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the KMNIST dataset
logger.info("Loading the train-test datasets...")
# Define the path where the KMNIST dataset is stored
DATASET_PATH = 'data/KMNIST/raw'

# Check if the dataset is already downloaded
if os.path.exists(DATASET_PATH):
    logger.info("KMNIST dataset already exists. Loading it directly...")
    download = False  # No need to download again
else:
    logger.info("KMNIST dataset not found. Downloading...")
    download = True  # Download if not found

trainData = KMNIST(root='data', train=True, download=download,
                   transform=ToTensor())
testData = KMNIST(root='data', train=False, download=download,
                  transform=ToTensor())
logger.info("Loaded the train-test datasets")

# Calculate the train-val split
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
                                    [numTrainSamples, numValSamples],
                                    generator=torch.Generator().manual_seed(42))
logger.info("Split the training data to train-val split")

# Initialise data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
                             batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, shuffle=True,
                           batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, shuffle=True,
                            batch_size=BATCH_SIZE)
logger.info(f"Initialised data loaders")

# Calculate steps per epoch
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# Initialise the LeNet model
logger.info("Initialising the LeNet model...")
model = LeNet(num_channels=1,
              classes=len(trainData.dataset.classes)).to(device)

# initialise the optimiser and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# Initialise a dictionary to store training history
H = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Noting the starting time
logger.info("Training the network...")
startTime = time.time()

# looping over epochs
for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()

    totalTrainLoss = 0
    totalValLoss = 0

    trainCorrect = 0
    valCorrect = 0

    # Progress bar for training loop
    trainLoader = tqdm(trainDataLoader, desc=f"Epoch {e+1}/{EPOCHS} - Training", leave=False)

    # loop over the train set
    for (x, y) in trainLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))

        # Perform a forward pass and calculate training loss
        pred = model(x)
        loss = lossFn(pred, y)

        # zero out the gradients, perform backpropagation
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()

    # Switch off autograd for evaluation
    with torch.no_grad():
        # set the model to eval mode
        model.eval()

        # Progress bar for validation loop
        valLoader = tqdm(valDataLoader, desc=f"Epoch {e+1}/{EPOCHS} - Validation", leave=False)

        # loop over the val set
        for (x, y) in valLoader:
            (x, y) = (x.to(device), y.to(device))

            # Make predictions and calculate val loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)

            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

    # Calculate average train-val loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # Calculate train-val accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)

    H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
    H['train_acc'].append(trainCorrect)
    H['val_loss'].append(avgValLoss.cpu().detach().numpy())
    H['val_acc'].append(valCorrect)

    # Log the model training and validation information
    logger.info(f'EPOCH:{e + 1}/{EPOCHS}')
    logger.info(f'Train loss:{avgTrainLoss:.6f}, Train accuracy:{trainCorrect:.4f}')
    logger.info(f'Val loss:{avgValLoss:.6f}, Val accuracy:{valCorrect:.4f}')

endTime = time.time()
logger.info(f'Total time taken to train them model: {endTime - startTime:.2f}s')

# Evaluate the network
logger.info(f'Evaluating the network...')

with torch.no_grad():
    model.eval()

    preds = []

    for (x, y) in testDataLoader:
        x = x.to(device)

        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

    # Generate classification report
    print(classification_report(testData.targets.cpu().numpy(),
                                np.array(preds), target_names=testData.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# Save the plot
if not os.path.exists('output'):
    os.makedirs('output')

plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])

logger.info("train.py has run successfully")
