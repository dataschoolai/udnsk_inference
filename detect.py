#Import image processing library
import cv2
#Import image processing library: PIL
from PIL import Image
import numpy as np
#Import libarary for file and directory handling
from pathlib import Path
#Import library for machine learning
import torch
from tqdm import tqdm
#Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print device
print("Device: ", device)