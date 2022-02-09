#Import image processing library
import cv2
#Import image processing library: PIL
from PIL import Image
import numpy as np
#Import libarary for file and directory handling
from pathlib import Path
import shutil
#Import library for machine learning
import torch
from tqdm import tqdm
import json
import argparse

#Define the function to create output directory
def make_output_dir(output_dir:str):
    """
    Create output directory if it does not exist
    Parameters:
    output_dir(str): path to output directory
    Returns:
    path(Path): path to output directory
    """
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)
    return path
    

    
#Define the function to predict image
def predict(input_dir:Path,output_dir:Path,model_path:Path):
    """
    Predict images in input directory
    Parameters:
    input_dir(Path): path to input directory
    output_dir(Path): path to output directory
    Returns:
    None
    """

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    #Create a list of all images_list in the directory
    images_list = [x for x in input_dir.glob('*.jpg')]
    #loop through all images_list
    for path in tqdm(images_list):
        #Open image:PIL
        image = Image.open(path)
        # image = cv2.imread(str(path))
        #Get prediction from model
        prediction = model(image)
        preds = []
        #Get bounding boxes
        if prediction.pred[0].shape[0]:
            # list of tensors pred[0] = (xyxy, conf, cls)
            for pred in prediction.pred[0]:
                #Get bounding box
                x1, y1, x2, y2 = pred[:4].tolist()
                #Get confidence
                conf = pred[4]
                #Get class
                cls = int(pred[5])
                #Get class name
                # cls_name = classes[cls]
                #Append to list
                preds.append([x1, y1, x2, y2, float(conf)])
        
        image_name = path.stem
        pred_label = prediction.names[0]
        #Save prediction to file in JSON format
        output_file = output_dir / Path(image_name).with_suffix(suffix='.json') 
        with open(output_file, 'w') as f:
            json.dump({'label':pred_label,'preds':preds}, f)



#Input directory
# INPUT_DIR = Path.cwd()/'data/104'
# OUTPUT_DIR = make_output_dir('output')

#Call the function to predict images

#Define the main function
def main():
    """
    Main function
    Parameters:
    None
    Returns:
    None
    """

    #Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print device
    print("Device: ", device)


    #Get input and output directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
    #Get model path
    parser.add_argument('--model_dir', type=str, required=True, help='path to model')
    args = parser.parse_args()

    #Call the function to predict images
    OUTPUT_DIR = make_output_dir(args.output_dir)
    INPUT_DIR =  Path.cwd() / args.input_dir


    #Create weights veriable and assign path of weights model
    MODEL_PATH = Path.cwd()/ args.model_dir
    print("Path of weights: ", MODEL_PATH)

    print("Input directory: ", INPUT_DIR)
    predict(INPUT_DIR,OUTPUT_DIR,MODEL_PATH)


if __name__ == '__main__':
    main()

    