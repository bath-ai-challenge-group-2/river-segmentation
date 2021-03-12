import keras
from keras import layers
from keras import models

class threshold_autoencoder:
    def __init__(self):
        self.no_models = 3
        self.autoencoders = None
        self.n_epochs = 5
        self.X_train = None
        
    
    def autoencoder_def(self):
        input = Input(shape=(256, 256, 1))
        
        x = layers.Conv2D(16, (1, 1), activation='relu', padding='same',kernel_regularizer = regularizers.l2(0.01))(input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (1, 1), activation='relu', padding='same',kernel_regularizer = regularizers.l2(0.01))(x)
        
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        
        x = layers.Conv2D(32, (1, 1), activation='relu', padding='same',kernel_regularizer = regularizers.l2(0.01))(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (1, 1), activation='relu', padding='same',kernel_regularizer = regularizers.l2(0.01))(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same',kernel_regularizer = regularizers.l2(0.01))(x)
        
        autoencoder = keras.Model(input, decoded)
        autoencoder.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy')
        
        return autoencoder
    
    def fit(self, X,y, X_test, y_test):
        autoencoder_list = []
        self.X_train = X
        for i in range(self.no_models):
            autoencoder = self.autoencoder_def()
            for n in range(self.n_epochs):
                plt.imsave(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\models_ae\trial_%s_%s.jpg'%(i,n), self.X_train[0].reshape(256,256))
                autoencoder.fit(self.X_train, y, epochs = 1, batch_size=8, shuffle=True, validation_data=(X_test, y_test))
                test = autoencoder.predict(X_ae[0].reshape(1,256,256,1))
                plt.imsave(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\trial maps\trial_%s_%s.jpg'%(i,n), test.reshape(256,256))
    
            preds = autoencoder.predict(self.X_train)
            
            for j in range(len(preds)):
                preds[j] = np.where(preds[j] > np.mean(preds[j]),1,0)
                
            self.X_train = preds
            
            if i < self.no_models:
                autoencoder_list.append(autoencoder)
            
        self.autoencoders = autoencoder_list
        
        return
    
    def predict(self,X):
        X_data = X
        for i in range(self.no_models):
            autoencoder = self.autoencoders[i]
            pred = autoencoder.predict(X)
            pred = np.where(pred > np.mean(pred) + np.std(pred[i]),1,0)
            X_data = pred
        
        return X_data.reshape(len(X_data),256,256)
            
            
ae = threshold_autoencoder()          
ae.fit(X_train, y_train, X_test, y_test)  
test = ae.predict(X_ae)            
            
            
        
n_epochs = 20

for i in range(n_epochs):
    print('Epoch No:', i)
    
    autoencoder.save(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\models_ae\model_%s.h5'%(i))
    test = autoencoder.predict(X_ae[0].reshape(1,256,256,1))
    plt.imsave(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\trial maps\%s.jpg'%i, test.reshape(256,256))
    
