import cv2
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset 
import torch
import torchvision.models as models
import PIL.Image
import PIL
import string
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

class ApplyCLAHE:
    """
    Konwersja RGB → YUV, zastosowanie CLAHE na kanale Y, powrót do RGB.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        # Konwersja PIL.Image → numpy (RGB, uint8)
        img_np = np.array(img)
        # RGB → YUV
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        # CLAHE na kanale Y
        y_channel = img_yuv[:, :, 0]
        img_yuv[:, :, 0] = self.clahe.apply(y_channel)
        # YUV → RGB
        img_rgb_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return PIL.Image.fromarray(img_rgb_eq)


class Blur3x3:
    """
    Gaussian Blur z kernelem 3×3 (OpenCV).
    """
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, ksize=(3, 3), sigmaX=0)
        return PIL.Image.fromarray(blurred)


class AddGaussianNoise:
    """
    Dodaje gaussowski szum do tensora (po ToTensor).
    """
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"