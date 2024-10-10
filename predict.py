import numpy as np

np.random.seed(42)

import logging
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2

# Setting up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to the trained PyTorch model")
args = vars(ap.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load KMNIST dataset and randomly grab 10 datapoints
logger.info("Loading the KMNIST test dataset")
testData = KMNIST(root="data", train=False, download=True,
                  transform=ToTensor())
idxs = np.random.choice(range(len(testData)), size=(10,))
testData = Subset(testData, idxs)

# Initialise the test dataloader
testDataLoader = DataLoader(testData, batch_size=1)

# Load the model and set it to eval mode
model = torch.load(args['model'], weights_only=False).to(device)
model.eval()

with torch.no_grad():
    for (image, label) in testDataLoader:
        origImg = image.numpy().squeeze(axis=(0, 1))
        gtLabel = testData.dataset.classes[label.numpy()[0]]

        image = image.to(device)
        pred = model(image)

        idx = pred.argmax(axis=1).cpu().numpy()[0]
        predLabel = testData.dataset.classes[idx]

        # Display the original image alongside the prediction
        origImg = np.dstack([origImg] * 3)
        origImg = imutils.resize(origImg, width=128)

        color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
        cv2.putText(origImg, gtLabel, (2, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

        print(f'Ground truth label: {gtLabel}, predicted label: {predLabel}')
        cv2.imshow("image", origImg)
        cv2.waitKey(0)


logger.info("predict.py has run successfully")
