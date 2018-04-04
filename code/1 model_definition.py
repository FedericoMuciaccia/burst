
# Copyright (C) 2018  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras


height, width, channels = [64, 256, 3] # TODO hardcoded
number_of_classes = 2 # TODO hardcoded

# model definition

# NOTA: per la profondità della rete comanda il lato corto dell'immagine, che è di 64
# immagini 64x256 pixels, quindi 6 blocchi convolutivi (2^6=64 level=6)

# TODO provare a fare una rete puramente convolutiva, senza max pooling e flatten e fully connected finali (perché può essere resa una rete generativa)
model = keras.models.Sequential() # TODO model functional API e layer keras.Input

model.add(keras.layers.ZeroPadding2D(input_shape=[height, width, channels]))
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')) # TODO check initializers a tutti i layer
model.add(keras.layers.Activation('relu')) # TODO vedere maxout
#keras.layers.normalization.BatchNormalization # TODO
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')) # TODO valutare pooling a 3 parzialmente interallacciato # TODO perché qui avevo messo padding='same'?
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')) # TODO vs 'same' (vedere ultimo layer)
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Convolution2D(filters=8, kernel_size=3, strides=1, padding='valid', use_bias=True))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
model.add(keras.layers.Dropout(rate=0.1))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=number_of_classes, use_bias=True)) # TODO check initializers
model.add(keras.layers.Activation('softmax'))

# write the model summary on a file
import sys
old_stdout = sys.stdout
summary_file = open('/storage/users/Muciaccia/burst/models/model_summary.txt', 'w')
sys.stdout = summary_file
model.summary() # goes to the standard output
summary_file.close()
sys.stdout = old_stdout

model.summary() # TODO vedere se si riesce a scriverlo su file tramite la nuova sintassi della funzione print di python
print('number of parameters:', model.count_params())

# model compiling
model.compile(loss='categorical_crossentropy',
	          optimizer=keras.optimizers.Adam(),
	          metrics=['accuracy']) # 'categorical_accuracy', 'precision', 'recall'

# save untrained model
model.save('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')
# (saving the whole model: architecture + weights + training configuration + optimizer state)



