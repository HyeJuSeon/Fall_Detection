import os
import json
import numpy as np
import cv2
import pickle

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res

def read_json(path):
    with open(path, 'r') as file:
        json_file = json.load(file)
    return json_file

def write_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_image(image):
    img_array = np.fromfile(image, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
