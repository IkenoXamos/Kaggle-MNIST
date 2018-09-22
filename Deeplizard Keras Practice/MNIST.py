import pandas as pd
import numpy as np

test = pd.read_csv('Deeplizard Keras Practice/test.csv')
train = pd.read_csv('Deeplizard Keras Practice/train.csv')

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


from sklearn.metrics import confusion_matrix
import itertools


X_train = train.drop(labels=['label'], axis = 1)
Y_train = train['label']

Y_train = to_categorical(Y_train, 10)
X_train = X_train / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test / 255.0
test = test.values.reshape(-1, 28, 28, 1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)




model = Sequential()

model.add(Conv2D(32, (5,5), strides = (1,1), \
    input_shape = (28, 28, 1), data_format = 'channels_last', padding = 'same', activation = 'relu'))
model.add(Conv2D(32, (5,5), strides = (1,1), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dense(10, activation='softmax'))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 120), validation_data = (X_val, Y_val), steps_per_epoch = X_train.shape[0] / 120, epochs=30, verbose=2, callbacks = [learning_rate_reduction])


model.save('MNIST weights.h5')


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))



results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_submission.csv",index=False)