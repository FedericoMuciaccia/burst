
# Copyright (C) 2017  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras

import h5py
import numpy
import sklearn
import sklearn.utils
import sklearn.preprocessing
import pandas

level = 6

signal_to_noise_ratio = 50

signal_file_path = '/storage/users/Muciaccia/burst/data/test_g_modes/SNR_{}/level_{}.hdf5'.format(signal_to_noise_ratio, level)
signal_images = h5py.File(signal_file_path)['spectro']
number_of_samples, height, width, channels = signal_images.shape
signal_classes = numpy.ones(number_of_samples)
noise_file_path = '/storage/users/Muciaccia/burst/data/gaussian_white_noise/level_{}.hdf5'.format(level)
noise_images = h5py.File(noise_file_path)['spectro'][slice(number_of_samples)]
noise_classes = numpy.zeros(number_of_samples)
images = numpy.concatenate([signal_images, noise_images])
classes = numpy.concatenate([signal_classes, noise_classes])

# data shuffle
images, classes = sklearn.utils.shuffle(images, classes)

#from matplotlib import pyplot
#
#from sklearn.model_selection import train_test_split

#height = 2**level # 64 frequency divisions
#width = 256 # time bins
#channels = 3 # number of detectors

number_of_classes = 2 # 4
to_categorical = sklearn.preprocessing.OneHotEncoder(n_values=number_of_classes, sparse=False, dtype=numpy.float32)
classes = to_categorical.fit_transform(classes.reshape(-1,1)) # TODO

# model definition
model = keras.models.Sequential() # TODO model functional API e layer keras.Input

model.add(keras.layers.ZeroPadding2D(input_shape=[height, width, channels]))
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True)) # TODO check initializers
model.add(keras.layers.Activation('relu'))
#keras.layers.normalization.BatchNormalization
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')) # TODO vs 'valid' (vedere ultimo layer)
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=number_of_classes, use_bias=True)) # TODO check initializers
model.add(keras.layers.Activation('softmax'))

model.summary()
print('number of parameters:', model.count_params())

# model compiling
model.compile(loss='categorical_crossentropy',
	          optimizer=keras.optimizers.Adam(),
	          metrics=['accuracy']) # 'categorical_accuracy', 'precision', 'recall'

# save untrained model
model.save('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')

# train parameters
number_of_epochs = 50
minibatch_size = 64

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

# model training and testing
history = model.fit(images, classes,
	batch_size=minibatch_size,
	epochs=number_of_epochs,
	verbose=True,
	#validation_data=(validation_images, validation_classes),
	validation_split=0.5,
	shuffle=True,
	callbacks=[early_stopping]) #  train data shuffled at each epoch. validation data never shuffled

# save train history
train_history = pandas.DataFrame(history.history) # TODO mettere colonne
train_history.to_csv('/storage/users/Muciaccia/burst/models/training_history.csv', index=False)

# save trained model
model.save('/storage/users/Muciaccia/burst/models/trained_model.hdf5')

################################

model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model.hdf5')


