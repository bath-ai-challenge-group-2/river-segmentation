from keras.optimizers import RMSprop
from keras.models import load_model
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from matplotlib import pyplot
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout, concatenate, Input, Flatten, Dense
from keras.models import Model
from keras import regularizers
from sklearn import metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from numpy import random
from keras import backend as K
import keras
from keras import layers


def models(with_auto):
    
    input = Input((256,256,3))
    conv1 = Conv2D(32, (1,1), activation='relu')(input)
    pool1 = MaxPooling2D(2, 2)(conv1) 
    conv2 = Conv2D(64, (1,1), activation='relu')(pool1)
    pool2 = MaxPooling2D(2, 2)(conv2)
    encoder = Model(inputs=[input],outputs = [pool2])
    encoder.compile(loss='binary_crossentropy', optimizer=RMSprop(lr = 1e-4) , metrics=['accuracy'])
    
    input_class = Input((K.int_shape(pool2)[1],K.int_shape(pool2)[2],K.int_shape(pool2)[3]))
    flat1 = Flatten()(input_class)
    dense1 = Dense(256, activation='relu')(flat1)
    dropout1 = Dropout(0.5)(dense1)
    out_class = Dense(1, activation='sigmoid')(dropout1)
    class_model = Model(inputs=[input_class],outputs = [out_class])
    
    input_pool2 = Input((K.int_shape(pool2)[1],K.int_shape(pool2)[2],K.int_shape(pool2)[3]))
    conv3 = Conv2D(64, (1,1), activation='relu',kernel_regularizer = regularizers.l2(0.01))(input_pool2)
    up1 = layers.UpSampling2D((2, 2))(conv3)
    conv4 = Conv2D(32, (1,1), activation='relu',kernel_regularizer = regularizers.l2(0.01))(up1)
    up2 = layers.UpSampling2D((2, 2))(conv4)
    decoded = Conv2D(1, (1,1), activation='sigmoid')(up2)
    decoder = Model(inputs = [input_pool2], outputs = [decoded])
    
    if with_auto == True:
        return encoder, decoder
    
    else:
         return encoder, class_model

def class_model(encoder, classification):
    
    classifier = Sequential()
    classifier.add(encoder)
    classifier.add(classification)
    classifier.compile(loss='binary_crossentropy', optimizer=RMSprop(lr = 1e-3) , metrics=['accuracy'])
    return classifier    

def auto_model(encoder, decoder): 
    
    encoder.trainable = False
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-4))
    return model 

encoder, classification = models(False)
classifier = class_model(encoder, classification)
classifier.fit(X_train, y_train, batch_size = 8 ,epochs = 5, verbose=1, validation_data=(X_test, y_test))

_, decoder = models(True)
autoencoder = auto_model(encoder, decoder)
autoencoder.fit(X_train_ae, y_train_ae, batch_size = 16 ,epochs = 10, verbose=1, validation_data=(X_test_ae, y_test_ae))

test = autoencoder.predict(X_test_ae[1].reshape(1,256,256,3))
test = np.where(test > np.mean(test),1,0)

plt.imshow(test.reshape(256,256))
plt.imshow(y_test_ae[1].reshape(256,256))

best_model.predict(X[1].reshape(1,256,256,3))