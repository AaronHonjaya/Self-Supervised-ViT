import cv2
import numpy as np
import os
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
import shutil

path = "./tiny-imagenet-200"

classes = set()



with open(os.path.join(path, "wnids.txt"), 'r') as file:
    for line in file:
        classes.add(line.strip())




wnid_to_words = SortedDict()
with open(os.path.join(path, "words.txt"), 'r') as file:
    for line in file:
        
        splits = line.strip().split("\t")
        
        wnid, words = splits[0], splits[1]
        if wnid in classes:
            wnid_to_words[wnid] = words

with open(os.path.join(path, "classes.txt"), "w+") as file:
    for wnid, words in wnid_to_words.items():
        file.write(f'{wnid}\t{words}\n')


def save_image(img_path, save_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    npy_image = np.array(image).astype(np.float32)
    
    npy_image /= 255.0
    npy_image = npy_image.transpose(2, 0, 1)
    
    np.save(save_path, npy_image)
    
    
    

    
# ******************************************** #   
# NOTE: Uncomment this to load images to numpy
# ******************************************** #   

# # train data to numpy
# train_path = os.path.join(path, "train")
# for subdir in os.listdir(train_path):
#     images_dir = os.path.join(train_path, subdir, "images")
#     npz_dir = os.path.join(train_path, subdir, "images_npy")
#     if not os.path.isdir(npz_dir):
#             os.mkdir(npz_dir)
    
#     for file in os.listdir(images_dir):
#         file = os.path.splitext(file)[0]
#         save_image(os.path.join(images_dir, file), os.path.join(npz_dir, file))          
    
  
  
# # test data to numpy
# test_path = os.path.join(path, "test")
# npz_dir = os.path.join(test_path, "images_npy")
# if not os.path.isdir(npz_dir):
#     os.mkdir(npz_dir)    
        
# for file in os.listdir(os.path.join(test_path, "images")):
#     file = os.path.splitext(file)[0]

#     img_path = os.path.join(test_path, "images", file)
#     save_path = os.path.join(npz_dir, file)
#     save_image(img_path, save_path)
    
 

