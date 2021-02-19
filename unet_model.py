from keras_unet.models import custom_unet
from keras import backend as K
from keras.optimizers import Adam 
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate, Input
from keras.models import Model
from keras import regularizers
from sklearn import metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numpy import random

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +smooth)

def dice_coef_loss(y_true, y_pred):
    print("dice loss")
    return 1-dice_coef(y_true, y_pred)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def Down_block(x,filters,kernal_size,strides,pool_size,dropout,activation,regularizer):
    x_1 = Conv2D(filters,kernal_size,strides = strides,activation=activation,kernel_regularizer = regularizer,padding = 'same')(x)
    x_2 = Conv2D(filters,kernal_size,strides =strides,activation=activation,kernel_regularizer = regularizer,padding = 'same')(x_1)
    x_3 = MaxPooling2D(pool_size = pool_size)(x_2)
    x_4 = Dropout(dropout)(x_3)
    return x_4, x_2

def Middle_block(x,filters,kernal_size,strides, activation, regularizer):
    x_1 = Conv2D(filters,kernal_size,strides = strides,activation=activation, kernel_regularizer = regularizer,padding = 'same')(x)
    x_2 = Conv2D(filters,kernal_size,strides = strides,activation=activation, kernel_regularizer = regularizer,padding = 'same')(x_1)
    return x_2
    
def Up_block(x,y,filters,kernal_size,strides,pool_size,dropout, activation, regularizer):
    x_1 = Conv2D(filters,kernal_size,strides = strides,activation=activation, kernel_regularizer = regularizer,padding = 'same')(UpSampling2D(size = (2,2))(x))
    x_2 = concatenate([y,x_1], axis = 3)
    x_3 = Dropout(dropout)(x_2)
    x_4 = Conv2D(filters, kernal_size,strides =strides, activation=activation, kernel_regularizer = regularizer, padding='same')(x_3)
    x_5 = Conv2D(filters, kernal_size,strides =strides, activation=activation, kernel_regularizer = regularizer, padding='same')(x_4)
    return x_5
    
def Output(x,filters,kernal_size, activation):
    x_1 = Conv2D(filters,kernal_size,activation=activation)(x)
    return x_1

def unet_model_def(reg,dropout):
    
    input = Input((256,256,3))
    out1, conv1 = Down_block(input, 32, 1, (1,1), 2, dropout, 'relu',regularizers.l2(reg))
    out2, conv2 = Down_block(out1, 64, 1, (1,1), 2, dropout, 'relu',regularizers.l2(reg))
    out3, conv3 = Down_block(out2, 128, 1, (1,1), 2, dropout, 'relu',regularizers.l2(reg))
    out4, conv4 = Down_block(out3, 256, 1, (1,1), 2, dropout, 'relu',regularizers.l2(reg))
    out5, conv5 = Down_block(out4, 512, 1, (1,1), 2, dropout, 'relu',regularizers.l2(reg))
    
    out6 = Middle_block(out5, 1024, 1, (1,1),'relu',regularizers.l2(reg))
    
    out7 = Up_block(out6,conv5, 512, 1, (1,1),(2,2), dropout, 'relu',regularizers.l2(reg))
    out8 = Up_block(out7,conv4, 256, 1, (1,1),(2,2), dropout, 'relu',regularizers.l2(reg))
    out9 = Up_block(out8,conv3 ,128, 1, (1,1),(2,2), dropout, 'relu',regularizers.l2(reg))
    out10 = Up_block(out9,conv2, 64, 1, (1,1),(2,2), dropout, 'relu',regularizers.l2(reg))
    out11 = Up_block(out10,conv1, 32, 1, (1,1),(2,2), dropout, 'relu',regularizers.l2(reg))
    out12 = Output(out11,1,1,'sigmoid')
    model = Model(inputs=[input],outputs = [out12])
    
    return model


def hyper_tuning():
    
    for i in range(10):
        #l2 = np.linspace(0.001,0.1, 11)
        dropout = np.linspace(0.1,0.5, 5)
        
        rand2 = random.randint(6)
        
        model = unet_model_def(0.1, dropout[rand2])
        model.compile(optimizer=Adam(lr = 1e-4), loss= dice_coef_loss,  metrics = [auc])
    
        history = model.fit(X_new,y_new, batch_size = 8 ,epochs = 1, verbose=1, validation_data=(X_test, y_test))
    
        image_no = 1
        test_image = X_test[image_no]
        test_mask = y_test[image_no].reshape(256,256)
        test_image_reshaped = test_image.reshape(1,256,256,3)
        test = model.predict(test_image_reshaped)
        test = test.reshape(256,256)
        
        file_name = r'DO(%s).jpg'%(dropout[rand2])
        plt.imsave(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\trial maps\%s'%file_name, test)

model = unet_model_def(0.03, 0.14)
model.compile(optimizer=Adam(lr = 1e-4), loss= dice_coef_loss,  metrics = [auc])
    
history = model.fit(X_new,y_new, batch_size = 8 ,epochs = 5, verbose=1, validation_data=(X_test, y_test))
       
hyper_tuning()

image_no = 2
test_image = X_test[image_no]
test_mask = y_test[image_no].reshape(256,256)
test_image_reshaped = test_image.reshape(1,256,256,3)
test = model.predict(test_image_reshaped)
test = test.reshape(256,256)
av = np.mean(test)
std = np.std(test)
test[test > (av)] = 1
test[test < (av)] = 0

plt.imshow(test_image)
plt.imshow(test_mask)
plt.imshow(test)