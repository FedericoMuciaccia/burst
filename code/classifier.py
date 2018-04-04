
# Copyright (C) 2017 Federico Muciaccia (federicomuciaccia@gmail.com)
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
import h5py # TODO vedere se persiste la costrizione di importarlo dopo keras
import numpy
import sklearn
import sklearn.utils
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics
import pandas

import matplotlib
from matplotlib import pyplot
matplotlib.rcParams.update({'font.size': 25}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

# TODO fare anche rete generativa, che partendo da random noise massimizza il neurone finale di segnale, per controllare visivamente che la rete abbia capito di cosa stiamo parlando. fare la stessa cosa anche per i vari kernel

#########################

# data loading


signal_to_noise_ratio = 40 # 40 35 30 25 20 15 10


dataset = h5py.File('/storage/users/Muciaccia/burst/data/preprocessed/SNR_{}.hdf5'.format(signal_to_noise_ratio))

train_images = dataset['train_images']
train_classes = dataset['train_classes']
test_images = dataset['test_images']
test_classes = dataset['test_classes']

#########################

# model training

model = keras.models.load_model('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')
# TODO il primo training è abbastanza dificile (lungo) se lo si fa coi i dropout, quindi penso che potrebbe valere la pena eliminarli soltanto per la prima tornata di dati, in modo da avvicinarsi molto al minimo

minibatch_size = 64 # TODO valutare se metterlo a 128 per avere un po' più di statistica e stabilità del training

iterations = {'SNR_40':  9000, # (avendo cura di controllare la corretta veloce convergenza nelle prime epoche)
              'SNR_35':  6000,
              'SNR_30':  5000,
              'SNR_25':  6000,
              'SNR_20': 10000,
              'SNR_15': 12000,
              'SNR_10': 13000}

number_of_iterations = iterations['SNR_{}'.format(signal_to_noise_ratio)] # TODO il numero cambia a seconda dell'SNR e della profondità di addestramento che si vuole raggiungere
cumultative_number_of_train_images = number_of_iterations * minibatch_size
dataset_size = len(train_images)
number_of_epochs = numpy.ceil(cumultative_number_of_train_images / dataset_size).astype(int)

# train parameters
#number_of_epochs = 25 # TODO forse è meglio farlo in numero di interazioni, dato che a seconda dell'SNR il numero di immagini è diverso e dunque lo è la lunghezza di una singola epoca

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')

class IterationHistory(keras.callbacks.Callback):
    '''
    callback to monitor the metrics at every iteration
    '''
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []
        # TODO non ci sono le quantità di testing
    
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        # TODO non ci sono le quantità di testing

iteration_history = IterationHistory()

try:
    train_history = model.fit(train_images, train_classes,
	    batch_size=minibatch_size,
	    epochs=number_of_epochs,
	    verbose=True,
	    validation_data=(test_images, test_classes),
	    #validation_split=0.5,
	    shuffle=True, # train data shuffled at each epoch. validation data never shuffled
	    callbacks=[iteration_history]
	    #callbacks=[early_stopping] # TODO mettere callback per la visualizzazione interattiva su TensorBoard
	    # TODO far decrescere gradualmente il learning rate durante il curriculum learning
	    # TODO far scrivere a intervalli regolari il numero di iterazioni (tipo ogni 100 iterazioni, che corrispondono a 6400 immagini, per poi poter fare il grafico del curriculum learning)
	    )
except KeyboardInterrupt: # TODO fare in modo che venga comunque salvata la history
    print('\n')
    print('manual early stopping!') # TODO automatizzare, magari tramite un callback

# save the trained model
model.save('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))
# TODO oppure salvare solo i pesi, in modo da poter successivamente modificare l'entità del dropout

# save the train history
train_history = pandas.DataFrame(train_history.history)
train_history.to_csv('/storage/users/Muciaccia/burst/models/training_history_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)

# save the detailed train history
detailed_train_history = pandas.DataFrame({'loss': iteration_history.loss,
                                           'accuracy': iteration_history.accuracy})
detailed_train_history.index.name = 'iteration'
detailed_train_history.to_csv('/storage/users/Muciaccia/burst/models/detailed_training_history_SNR_{}.csv'.format(signal_to_noise_ratio), index=True)
# TODO non si possono avere le quantità di test per la singola iterazione ma solo per la singola epoca

################################

# model validation

model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))

predictions = model.predict(test_images, batch_size=128, verbose=1) # the minibatch size doesn't seem to influence the prediction time
predicted_signal_probabilities = predictions[:,1]
true_classes = test_classes[:,1]

threshold = 0.5 # TODO fine tuning ed istogramma
predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)

# TODO salvare i valori di predicted_signal_probabilities e predicted_classes per poter fare i grafici velocemente in un secondo momento

is_correctly_predicted = numpy.equal(predicted_classes, true_classes)
misclassified_images = test_images[numpy.logical_not(is_correctly_predicted)]
misclassified_classes = true_classes[numpy.logical_not(is_correctly_predicted)]

print('misclassified images:',len(misclassified_images))
# TODO far scivere gli indici (pre-shuffle) delle immagini misclassificate su un file separato per ogni SNR

def view_image(image):
    pyplot.imshow(image, interpolation='none', origin="lower")
    pyplot.show()
    #pyplot.savefig('example.jpg') # TODO levare bordo bianco
    pyplot.close()

for image in misclassified_images[slice(0,5)]: # print only 5 images
    view_image(image)

indices = numpy.arange(len(test_images))
misclassified_indices = indices[numpy.logical_not(is_correctly_predicted)]
# TODO attenzione alla operazioni di shuffle fatta in precedenza

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
           'false alarms (%)':100*false_positives/all_predicted_as_signals, #all_real_noises, # TODO capire meglio!!! (dovrebbe essere 1 - purity)
           'missed signals (%)':100*false_negatives/all_real_signals, # TODO il false dismissal è 1-efficiency ?
           'selected signals (%)':100*true_positives/all_real_signals,
           'purity (%)':100*purity,
           'efficiency (%)':100*efficiency,
           'accuracy (%)':100*accuracy}

results = pandas.DataFrame(metrics, index=[signal_to_noise_ratio])
results.to_csv('/storage/users/Muciaccia/burst/models/results_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)
# TODO poi concatenarli con pandas.concat()
# TODO magari levare le percentuali e rimettere le quantità normalizzate

# NOTA: ai fini della scoperta con 5 sigma di confidenza bisogna guardare il valore di purezza (o di false alarm)

# a mio avviso si sviluppa un leggero overfitting ad SNR 15 ed SNR 10

# TODO provare a diminiure gradualmente sia il dropout che il learning rate

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
ax1.set_xlabel('predicted probability to be in the "signal" class') # OR 'class prediction'
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
fig_predictions.savefig('/storage/users/Muciaccia/burst/media/classifier_output_SNR_{}.jpg'.format(signal_to_noise_ratio), bbox_inches='tight') 
pyplot.close()

#############################

# ROC curve

thresholds = numpy.linspace(start=0 ,stop=1 ,num=101) # 100 bins
thresholds = thresholds[1:-1] # to avoid problems at the borders

efficiency_list = []
false_alarm_list = []

for threshold in thresholds:
    predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)
    
    confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
    [[true_negatives,false_positives],[false_negatives,true_positives]] = [[predicted_0_true_0,predicted_1_true_0],[predicted_0_true_1,predicted_1_true_1]] = confusion_matrix
    
    efficiency = true_positives/(true_positives + false_negatives) # recall
    false_alarm = false_positives/(true_positives + false_positives)
    #purity = true_positives/all_predicted_as_signals # precision
    
    efficiency_list.append(efficiency)
    false_alarm_list.append(false_alarm)

efficiency_list = numpy.array(efficiency_list)
false_alarm_list = numpy.array(false_alarm_list)

ROC_curve = {#'SNR':signal_to_noise_ratio,
             #'level':level,
             #'all validation samples':all_validation_samples,
             'false_alarm':100*false_alarm_list,
             #'purity (%)':100*purity,
             'efficiency':100*efficiency_list}

ROC_curve = pandas.DataFrame(ROC_curve, index=thresholds)
ROC_curve.to_csv('/storage/users/Muciaccia/burst/models/ROC_curve_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)
# TODO magari levare le percentuali e rimettere le quantità normalizzate

