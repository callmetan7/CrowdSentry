import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class CrowdDataset(Dataset):
    def __init__(self, rootDir, transform=None, targetSize=(224, 224), outputSize=(112, 112)):
        self.rootDir = rootDir
        self.imagePaths = sorted(os.listdir(os.path.join(rootDir, "images")))
        self.densityPaths = sorted(os.listdir(os.path.join(rootDir, "density_maps")))
        self.transform = transform
        self.targetSize = targetSize
        self.outputSize = outputSize

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.rootDir, "images", self.imagePaths[idx])
        densityPath = os.path.join(self.rootDir, "density_maps", self.densityPaths[idx])
        image = cv2.imread(imgPath)
        density_map = np.load(densityPath)
        imageResized = cv2.resize(image, self.targetSize)
        densityMapResized = cv2.resize(density_map, self.outputSize)
        scalingFactor = (density_map.shape[0] / self.outputSize[0]) * (density_map.shape[1] / self.outputSize[1])
        densityMapResized *= scalingFactor

        imgTensor = torch.tensor(imageResized, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize image
        dmapTensor = torch.tensor(densityMapResized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return imgTensor, dmapTensor

