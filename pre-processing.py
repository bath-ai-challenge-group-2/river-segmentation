import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from numpy import asarray

# defining global variable path
path = r"C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\AI challenge data"

def loadImages(path):
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg')])
    mask_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.png')])
    
    return image_files, mask_files


def processing(data):
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data]
    print('Original size',img[0].shape)

    height = 256
    width = 256
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    return res_img

def normalise(data):
    
    dummy = []
    for image in data:
        
        image = image.astype('float32')
        means = image.mean(axis=(0,1), dtype='float64')
        image -= means
        mins = np.min(image,axis=(0,1))
        maxs = np.max(image, axis=(0,1))
        image = (image - mins) / (maxs - mins)
        dummy.append(image)
        
    return dummy

def to_array(data,image_type):
    if image_type == 'images':
        array = np.zeros([len(data),data[0].shape[0],data[0].shape[1], data[0].shape[2]])
        for i in range(len(data)):
            array[i] = data[i]
            
    if image_type == 'masks':
        array = np.zeros([len(data),data[0].shape[0],data[0].shape[1],2])
        for i in range(len(data)):
            array[i] = data[i].reshape(256,256,2)
    
    return array

def augment_image_data(X,y):
    dummy_X = np.zeros([len(X) * 4,X.shape[1],X.shape[2], X.shape[3]])
    dummy_y = np.zeros([len(y) * 4,y.shape[1],y.shape[2], 2])
    n = 0
    for i in range(len(X)):
        image = X[i]
        mask = y[i]
        for j in range(4):
            rot_image = np.rot90(image, j)
            rot_mask = np.rot90(mask, j)
            dummy_X[n] = rot_image
            dummy_y[n] = rot_mask
            n += 1
            
    return dummy_X, dummy_y

def augment_class_data(X,y):
    dummy_X = np.zeros([len(X) * 4,X.shape[1],X.shape[2], X.shape[3]])
    dummy_y = np.zeros([len(X) * 4])
    n = 0
    for i in range(len(X)):
        image = X[i]
        is_river = y[i]
        for j in range(4):
            rot_image = np.rot90(image, j)
            dummy_X[n] = rot_image
            dummy_y[n] = is_river
            n += 1
    
    return dummy_X, dummy_y

def class_data(images, masks):
    data = []
    is_river = []
    
    for image, mask in zip(images, masks):
        seg_image = mask.reshape(256,256,1) * image
        data.append(seg_image)
        is_river.append(1)
        
        anti_mask = np.where(mask == 1,0,1)
        anti_seg_image = anti_mask.reshape(256,256,1) * image
        data.append(anti_seg_image)
        is_river.append(0)
        
    return np.array(data), np.array(is_river)
        
def mask_prep(masks):
    dummy = []
    for mask in masks:
        new_mask = np.zeros([256,256,2])
        new_mask[:,:,0] = mask
        new_mask[:,:,1] = np.where(mask == 1,0,1)
        dummy.append(new_mask)
    return dummy
            
images, masks_list = loadImages(path)

res_images = processing(images)
res_images = normalise(res_images)
masks = processing(masks_list)
masks = mask_prep(masks)

X_ae = to_array(res_images, 'images')
y_ae = to_array(masks, 'masks')

X_ae, y_ae = augment_image_data(X_ae,y_ae)

test_image_path = [r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\test_image\test_image.jpg']

test_image  =  normalise(processing(test_image_path))[0]
     
            
        
    
