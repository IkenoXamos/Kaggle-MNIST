import pandas as pd
import numpy as np

test = pd.read_csv('Deeplizard Keras Practice\\test.csv')
train = pd.read_csv('Deeplizard Keras Practice\\train.csv')

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, load_model
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
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

model = load_model('MNIST weights.h5')

model.summary()

img = 4600
input_img = X_train[img].reshape(1, 28, 28, 1)
filter_index = 0
layer_name = 'conv2d_1'

plt.imshow(input_img.reshape(28,28), interpolation='nearest')
plt.show()
print("label : ", Y_train[img,:])

layerlist = ['conv2d_' + str(i + 1) for i in range(2)]



for layer_name in layerlist:
    model = load_model('MNIST weights.h5')
    
    model.outputs = [model.get_layer(layer_name).output]

    ActivationTensor = model.predict(input_img)
    
    for i in range(ActivationTensor.shape[3]):
        im = plt.imshow(ActivationTensor[:, :, :, i].reshape(28,28), interpolation='nearest')
        plt.savefig('Activations/Activation of ' + str(i) + 'th filter in layer ' + layer_name + '.png')
        plt.show()

layerlist = ['conv2d_' + str(i + 3) for i in range(2)]

for layer_name in layerlist:
    model = load_model('MNIST weights.h5')
    
    model.outputs = [model.get_layer(layer_name).output]

    ActivationTensor = model.predict(input_img)
    
    for i in range(ActivationTensor.shape[3]):
        im = plt.imshow(ActivationTensor[:, :, :, i].reshape(14,14), interpolation='nearest')
        plt.savefig('Activations/Activation of ' + str(i) + 'th filter in layer ' + layer_name + '.png')
        plt.show()

'''

from keras import backend as K

layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

grads = normalize(grads)

# this function returns the loss and grads given the input picture
iterate = K.function([model.input], [loss, grads])

img_width, img_height = 28, 28

step = 1.

# we start from a gray image with some random noise
input_img_data = np.random.random((1, img_width, img_height, 3))

input_img_data = (input_img_data - 0.5) * 20 + 128

# we run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

    print('Current loss value:', loss_value)
    if loss_value <= 0.:
        break

'''