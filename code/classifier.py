
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

# TODO fare anche rete generativa, che partendo da random noise massimizza il neurone finale di segnale, per controllare visivamente che la rete abbia capito di cosa stiamo parlando. fare la stessa cosa anche per i vari kernel

level = 6
# TODO fare diverse reti, una per livello, che collaborano nel prendere la decisione finale
# TODO o magari anche una rete che ha in input le probabilità date dalle singole reti ai vari livelli e decide globalmente il da farsi

signal_to_noise_ratio = 10 # 40 35 30 25 20 15 10

signal_file_path = '/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/level_{}.hdf5'.format(signal_to_noise_ratio, level)
signal_images = h5py.File(signal_file_path)['spectro']
signal_number_of_samples, height, width, channels = signal_images.shape

noise_file_path = '/storage/users/Muciaccia/burst/data/big_set_gaussian_white_noise/level_{}.hdf5'.format(level)
noise_images = h5py.File(noise_file_path)['spectro']
noise_number_of_samples, height, width, channels = noise_images.shape
# TODO randomizzare il campione di solo rumore ad ogni nuova generazione del dataset

# the two classes should be equipopulated
number_of_samples = numpy.min([signal_number_of_samples, noise_number_of_samples])

# # TODO
# # shuffle the noise data (before the selection)
# images, classes = sklearn.utils.shuffle(images, classes)
# # train-test split
# train_images, test_images, train_classes, test_classes = sklearn.model_selection.train_test_split(dataset.images, dataset.classes, test_size=0.5, shuffle=True)
# # TODO vedere nuova input pipeline di TensorFlow

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

number_of_classes = 2 # TODO 4
to_categorical = sklearn.preprocessing.OneHotEncoder(n_values=number_of_classes, sparse=False, dtype=numpy.float32)
classes = to_categorical.fit_transform(classes.reshape(-1,1)) # TODO

#########################

# model definition

# NOTA: comanda il lato corto dell'immagine, che è di 64
# immagini 64x256 pixels, quindi 6 blocchi convolutivi (2^6=64 level=6)

# TODO provare a fare una rete puramente convolutiva, senza max pooling e flatten e fully connected finali
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

model.summary() # TODO scriverlo su file, magari tramite la nuova sintassi della funzione print di python
print('number of parameters:', model.count_params())

# model compiling
model.compile(loss='categorical_crossentropy',
	          optimizer=keras.optimizers.Adam(),
	          metrics=['accuracy']) # 'categorical_accuracy', 'precision', 'recall'

# save untrained model
model.save('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')
# (saving the whole model: architecture + weights + training configuration + optimizer state)

####################

# model training

#model = keras.models.load_model('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')
# TODO il primo training è abbastanza dificile (lungo) se lo si fa coi i dropout, quindi penso che potrebbe valere la pena eliminarli soltanto per la prima tornata di dati, in modo da avvicinarsi molto al minimo

number_of_iterations = 5000 # TODO vedere 1024 o 2048 # TODO il numero cambia a seconda dell'SNR e della profondità di addestramento che si vuole raggiungere
minibatch_size = 64 # TODO valutare se metterlo a 128 per avere un po' più di statistica e stabilità del training

cumultative_number_of_train_images = number_of_iterations * minibatch_size
dataset_size = 2 * number_of_samples # noise and noise+signal
number_of_epochs = numpy.ceil(cumultative_number_of_train_images / dataset_size).astype(int)

# train parameters
#number_of_epochs = 25 # TODO forse è meglio farlo in numero di interazioni, dato che a seconda dell'SNR il numero di immagini è diverso e dunque lo èla lunghessa di una singola epoca

# SNR   epochs  iterations (with minibatch 64)
# ------------------------
# 40    25  
# 35    
# 30    
# 25    
# 20    15  
# 15    15      21300
# 10    25+     33500

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

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

# save trained model
model.save('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))
# TODO oppure salvare solo i pesi, in modo da poter successivamente modificare l'entità del dropout

# save train history
train_history = pandas.DataFrame(train_history.history) # TODO mettere colonne
train_history.to_csv('/storage/users/Muciaccia/burst/models/training_history_SNR_{}.csv'.format(signal_to_noise_ratio), index=False) # TODO vedere append della history (magari direttamente nel dataframe pandas) per curriculum learning

################################

# model validation

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
# TODO far scivere gli indici (pre-shuffle) delle immagini misclassificate su un file separato per ogni SNR

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
           'all validation samples':all_validation_samples,
           'misclassified images':false_positives+false_negatives,
           'false negatives':false_negatives,
           'false positives':false_positives,
           'rejected noise (%)':100*true_negatives/all_real_noises,
           'false alarms (%)':100*false_positives/all_real_noises,
           'missed signals (%)':100*false_negatives/all_real_signals,
           'selected signals (%)':100*true_positives/all_real_signals,
           'purity (%)':100*purity,
           'efficiency (%)':100*efficiency,
           'accuracy (%)':100*accuracy}

# TODO scrivere i risutati su file

# c'erano solo falsi negativi (segnali persi). nessun falso positivo
# NOTA: buono ai fini della scoperta con 5 sigma di confidenza


#{'SNR': 15,
# 'accuracy (%)': 99.927373563977284,
# 'all_validation_samples': 90876,
# 'efficiency (%)': 99.883357542145347,
# 'false alarms (%)': 0.028610414190765439,
# 'false_negatives': 53,
# 'false_positives': 13,
# 'level': 6,
# 'misclassified_images': 66,
# 'missed signals (%)': 0.1166424578546591,
# 'purity (%)': 99.97136437728534,
# 'rejected noise (%)': 99.971389585809234,
# 'selected signals (%)': 99.883357542145347}

#{'SNR': 10,
# 'accuracy (%)': 99.208698489651312,
# 'all_validation_samples': 85808,
# 'efficiency (%)': 98.885884765989189,
# 'false alarms (%)': 0.46848778668655605,
# 'false_negatives': 478,
# 'false_positives': 201,
# 'level': 6,
# 'misclassified_images': 679,
# 'missed signals (%)': 1.1141152340108149,
# 'purity (%)': 99.52846787247519,
# 'rejected noise (%)': 99.531512213313448,
# 'selected signals (%)': 98.885884765989189}


import matplotlib
# TODO svg engine
from matplotlib import pyplot

fig_predictions = pyplot.figure(figsize=[9,6])
ax1 = fig_predictions.add_subplot(111) # TODO
# predicted as noise
n, bins, rectangles = ax1.hist(predicted_signal_probabilities[true_classes == 0], 
    	                       bins=50,
    	                       range=(0,1),
    	                       #normed=True, 
    	                       histtype='step', 
    	                       #alpha=0.6,
    	                       color='#ff3300',
    	                       label='noise')
# predicted as signal+noise
n, bins, rectangles = ax1.hist(predicted_signal_probabilities[true_classes == 1], 
  	                           bins=50,
   	                           range=(0,1),
   	                           #normed=True, 
   	                           histtype='step', 
   	                           #alpha=0.6,
   	                           color='#0099ff',
   	                           label='noise + signal')
ax1.set_title('classifier output') # OR 'model output'
ax1.set_ylabel('count') # OR density
ax1.set_xlabel('predicted signal probability') # OR 'class prediction'
tick_spacing = 0.1
ax1.set_yscale('log')
ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))
#ax1.legend(loc='best')
#pyplot.axvline(x=best_threshold, # TODO
#	           color='grey', 
#	           linestyle='dotted', 
#	           alpha=0.8)
ax1.legend(loc='upper center')#, frameon=False)
#pyplot.show()
fig_predictions.savefig('/storage/users/Muciaccia/burst/media/classifier_output_SNR_{}.svg'.format(signal_to_noise_ratio), bbox_inches='tight') 
pyplot.close()




