
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
import sklearn.metrics
import pandas
from matplotlib import pyplot

level = 6
# TODO fare diverse reti, una per livello, che collaborano nel prendere la decisione finale
# TODO o magari anche una rete che ha in input le probabilità date dalle singole reti ai vari livelli e decide globalmente il da farsi

signal_to_noise_ratio = 15 # 40 35 30 25 20 15

signal_file_path = '/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/level_{}.hdf5'.format(signal_to_noise_ratio, level)
signal_images = h5py.File(signal_file_path)['spectro']
signal_number_of_samples, height, width, channels = signal_images.shape

noise_file_path = '/storage/users/Muciaccia/burst/data/big_set_gaussian_white_noise/level_{}.hdf5'.format(level)
noise_images = h5py.File(noise_file_path)['spectro']
noise_number_of_samples, height, width, channels = noise_images.shape
# TODO randomizzare il campione di solo rumore ad ogni nuova generazione del dataset

# TODO per aggirare la momentanea  possibilità che le immagini generate di segnale siano di più di quelle di noise (le due classi devono essere equipopolate)
number_of_samples = numpy.min([signal_number_of_samples, noise_number_of_samples])

signal_images = h5py.File(signal_file_path)['spectro'][slice(number_of_samples)] # TODO lentissimo
noise_images = h5py.File(noise_file_path)['spectro'][slice(number_of_samples)] # TODO lentissimo
signal_classes = numpy.ones(number_of_samples)
noise_classes = numpy.zeros(number_of_samples)

images = numpy.concatenate([signal_images, noise_images]) # TODO lentissimo # TODO vedere se la concatenazione comporta un inutile spreco del doppio della memoria
classes = numpy.concatenate([signal_classes, noise_classes])
# TODO vedere nuova pipeline standard di input per TensorFlow

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

#########################

# TODO provare a fare una rete puramente convolutiva, senza max pooling e flatten e fully connected finali

# model definition
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

model.summary()
print('number of parameters:', model.count_params())

# model compiling
model.compile(loss='categorical_crossentropy',
	          optimizer=keras.optimizers.Adam(),
	          metrics=['accuracy']) # 'categorical_accuracy', 'precision', 'recall'

# save untrained model
model.save('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')

# train parameters
number_of_epochs = 50 # TODO forse è meglio farlo in numero di interazioni, dato che a seconda dell'SNR il numero di immagini è diverso e dunque la lunghessa di una singola epoca
minibatch_size = 64 # TODO valutare se metterlo a 128 per avere un po' più di statistica e stabilità del training

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

# model training
try:
    train_history = model.fit(images, classes,
	    batch_size=minibatch_size,
	    epochs=number_of_epochs,
	    verbose=True,
	    #validation_data=(validation_images, validation_classes),
	    #validation_split=0.5,
	    shuffle=True, # train data shuffled at each epoch. validation data never shuffled
	    #callbacks=[early_stopping]
	    # TODO far decrescere gradualmente il learning rate durante il curriculum learning
	    # TODO far scrivere a intervalli regolari il numero di iterazioni (tipo ogni 100 iterazioni, che corrispondono a 6400 immagini, per poi poter fare il grafico del curriculum learning)
	    )
except KeyboardInterrupt: # TODO fare in modo che venga comunque salvata la history
    print('\n')
    print('manual early stopping!') # TODO automatizzare

# save train history
train_history = pandas.DataFrame(train_history.history) # TODO mettere colonne
train_history.to_csv('/storage/users/Muciaccia/burst/models/training_history.csv', index=False) # TODO vedere append della history per curriculum learning. oppure mettere history separate per i differenti SNR

# save trained model
model.save('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))

################################

# model testing

model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))

predictions = model.predict(images, batch_size=128, verbose=1) # the minibatch size doesn't seem to influence the prediction time
predicted_signal_probabilities = predictions[:,1]
true_classes = classes[:,1]

threshold = 0.5 # TODO fine tuning ed istogramma
predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)

is_correctly_predicted = numpy.equal(predicted_classes,true_classes)
misclassified_images = images[numpy.logical_not(is_correctly_predicted)]
misclassified_classes = true_classes[numpy.logical_not(is_correctly_predicted)]

print('misclassified images:',len(misclassified_images))

def view_image(image):
    pyplot.imshow(image, interpolation='none', origin="lower")
    pyplot.show()
    #pyplot.savefig('example.jpg', dpi=300)

for image in misclassified_images:
    view_image(image)

confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
[[true_negatives,false_positives],[false_negatives,true_positives]] = [[predicted_0_true_0,predicted_1_true_0],[predicted_0_true_1,predicted_1_true_1]] = confusion_matrix
purity = true_positives/(true_positives + false_positives) # precision
efficiency = true_positives/(true_positives + false_negatives) # recall
accuracy = (true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives)
# normalizations
all_real_signals = true_positives + false_negatives
all_real_noises = true_negatives + false_positives
all_predicted_as_signals = true_positives + false_positives
all_predicted_as_noise = true_negatives + false_negatives
all_validation_samples = true_positives + false_positives + true_negatives + false_negatives

metrics = {'SNR':signal_to_noise_ratio,
           'level':level,
           'all_validation_samples':all_validation_samples,
           'misclassified_images':false_positives+false_negatives,
           'false_negatives':false_negatives,
           'false_positives':false_positives,
           'rejected noise (%)':100*true_negatives/all_real_noises,
           'false alarms (%)':100*false_positives/all_real_noises,
           'missed signals (%)':100*false_negatives/all_real_signals,
           'selected signals (%)':100*true_positives/all_real_signals,
           'purity (%)':100*purity,
           'efficiency (%)':100*efficiency,
           'accuracy (%)':100*accuracy}

# ci sono solo falsi negativi (segnali persi). nessun falso positivo
# TODO buono ai fini della scoperta con 5 sigma di confidenza




#{'SNR': 15,
# 'accuracy (%)': 99.917254888716272,
# 'all_validation_samples': 59218,
# 'efficiency (%)': 99.83450977743253,
# 'false alarms (%)': 0.0,
# 'level': 6,
# 'misclassified_images': 49,
# 'missed signals (%)': 0.1654902225674626,
# 'purity (%)': 100.0,
# 'rejected noise (%)': 100.0,
# 'selected signals (%)': 99.834509777432544}





