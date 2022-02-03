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
device = torch.device("cpu")
#print device
print("Device: ", device)


#Create weights veriable and assign path of weights model
path_weights = Path.cwd()/ 'yolo_model/Exposure_100/weights/best.pt'
print("Path of weights: ", path_weights)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_weights)

#Input directory
input_dir = '/home/user/Documents/udnsk_inference/data/104'

# input_dir = '/media/buntuml/DATASET/DAMAGEAI/REPORT/spalling/DATASET_JSON/images_list'
#Create a list of all images_list in the directory
images_list = [x for x in Path(input_dir).glob('*.jpg')]
print(images_list[:10])
#loop through all images_list
for path in tqdm(images_list):
    #Open image:PIL
    image = Image.open(path)

    # image = cv2.imread(str(path))
    #Get prediction from model
    prediction = model(image)
    #Get bounding boxes
    if prediction.pred[0].shape[0]:
        prediction.files = [path.name]
        #Save prediction to file
        prediction.save('output')