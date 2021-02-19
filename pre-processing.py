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

def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

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
        array = np.zeros([len(data),data[0].shape[0],data[0].shape[1]])
        for i in range(len(data)):
            array[i] = data[i]
    
    return array

def augment_data(X,y):
    dummy_X = np.zeros([len(X) * 4,X.shape[1],X.shape[2], X.shape[3]])
    dummy_y = np.zeros([len(y) * 4,y.shape[1],y.shape[2], 1])
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
            
images, masks = loadImages(path)

res_images = processing(images)
y = processing(masks)

X = normalise(res_images)
X = to_array(X, 'images')
y = to_array(y, 'masks')
y = y.reshape((y.shape[0],256,256,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_new, y_new = augment_data(X_train,y_train)
            
            
        
    
