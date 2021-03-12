from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.models import load_model
import keras

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
x_train, x_test, y_train, y_test = train_test_split(X_ae, y_ae.reshape(len(y_ae),256,256,2), test_size=0.2)

# preprocess input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet', classes = 2, activation = 'sigmoid')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\models_ae\best_model.h5', save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau()
]

model.fit(x=x_train,y=y_train,batch_size=16,epochs = 40, callbacks = callbacks , validation_data=(x_test, y_test))

model = load_model(r'C:\Users\josep\OneDrive\Desktop\mres year programming\AI challenge\models_ae\best_model.h5', compile = False)

test = model.predict(test_image.reshape(1,256,256,3))
test = test.reshape(256,256,2)[:,:,0].round()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(test)
ax2.imshow(test_image)