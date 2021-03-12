import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.models import load_model
import keras

def preprocessing(data):
    img = cv2.imread(data, cv2.IMREAD_UNCHANGED)

    height = 256
    width = 256
    dim = (width, height)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    image = image.astype('float32')
    means = image.mean(axis=(0,1), dtype='float64')
    image -= means
    mins = np.min(image,axis=(0,1))
    maxs = np.max(image, axis=(0,1))
    image = (image - mins) / (maxs - mins)
    
    img_min = np.min(img,axis=(0,1))
    img_max = np.max(img,axis=(0,1))
    img = (img - img_min) / (img_max - img_min)
    
    return image, img

model = load_model(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\models_ae\best_model.h5', compile = False)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

image_path = r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\test_image\test_image.jpg'
test_image, original_image  =  preprocessing(image_path)

test = model.predict(test_image.reshape(1,256,256,3))
test = test.reshape(256,256,2)[:,:,0].round()
test = cv2.resize(test, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)

#for visualisation
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(test)
ax2.imshow(original_image)
ax3.imshow(original_image * test.reshape(original_image.shape[0], original_image.shape[1],1))
ax4.imshow(original_image * np.where(test.reshape(original_image.shape[0], original_image.shape[1],1) == 1, 0 ,1))
