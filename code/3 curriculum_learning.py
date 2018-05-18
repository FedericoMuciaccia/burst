
# Copyright (C) 2017 Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras
import h5py # TODO vedere se persiste la costrizione di importarlo dopo keras
import numpy
#import sklearn
#import sklearn.utils
#import sklearn.model_selection
#import sklearn.preprocessing
import sklearn.metrics
import pandas
import datetime

import config


# TODO fare anche rete generativa, che partendo da random noise massimizza il neurone finale di segnale, per controllare visivamente che la rete abbia capito di cosa stiamo parlando. fare la stessa cosa anche per i vari kernel

level = config.cWB_level

all_SNR = numpy.array(config.all_SNR)

for signal_to_noise_ratio in all_SNR:
    
    #signal_to_noise_ratio = 8
    print('SNR:', signal_to_noise_ratio)
    
    #########################
    
    # data loading
    
    dataset = h5py.File('/storage/users/Muciaccia/burst/data/preprocessed/SNR_{}.hdf5'.format(signal_to_noise_ratio))
    # TODO usare dask.array?
    
    # TODO greedy VS lazy?
    train_images = dataset['train_images'].value
    train_classes = dataset['train_classes'].value
    test_images = dataset['test_images'].value
    test_classes = dataset['test_classes'].value
    # TODO vedere la nuova input pipeline di TensorFlow
    
    #########################
    
    # model loading
    
    if signal_to_noise_ratio == all_SNR.max():
        model = keras.models.load_model('/storage/users/Muciaccia/burst/models/untrained_model.hdf5')
    else:
        indices = numpy.arange(len(all_SNR))
        current_index = indices[numpy.equal(all_SNR, signal_to_noise_ratio)]
        previous_index = current_index - 1
        previous_SNR = int(all_SNR[previous_index]) # TODO dovrebbero esserci dei banalissimi metodi .previous() e .next() # TODO ci sono sempre problemi nella conversione degli indici da numpy a python
        model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(previous_SNR))
    # TODO il primo training è abbastanza dificile (lungo) se lo si fa coi i dropout, quindi penso che potrebbe valere la pena eliminarli soltanto per la prima tornata di dati, in modo da avvicinarsi molto al minimo
    
    ##########################
    
    # model training
    
    minibatch_size = 64 # TODO valutare se metterlo a 128 per avere un po' più di statistica e stabilità del training
    
    iterations = {'SNR_40': 10000, # (avendo cura di controllare la corretta veloce convergenza nelle prime epoche, anche se spesso non corverge per le prime 10)
                  'SNR_35':  6000,
                  'SNR_30':  5000,
                  'SNR_25':  6000,
                  'SNR_20': 10000,
                  'SNR_15': 12000,
                  'SNR_12': 12000, # TODO
                  'SNR_10': 13000,
                  'SNR_8': 13000} # TODO
    
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
    
#    # TODO per poter poi fare il filmatino dell'evoluzione dell'istogramma durante l'addestramento
#    # TODO e per controllare se è possibile monitorare l'overfit direttamente tramite l'istogramma degli outut del solo dataset di train
#    class PredictionHistory(keras.callbacks.Callback):
#        '''
#        callback to monitor the predictions at every epoch
#        '''
#        def on_train_begin(self, logs={}):
#            self.train_predictions = []
#            self.test_predictions = []
#        
#        def on_epoch_end(self, epoch, logs={}):
#            self.train_predictions.append(self.model.predict(train_images, batch_size=128)[:,1]) # TODO riutilizzare quelle già computate
#            self.test_predictions.append(self.model.predict(test_images, batch_size=128)[:,1])
#        
#        def on_train_end(self, logs={}):
#            self.train_predictions = numpy.array(self.train_predictions)
#            self.test_predictions = numpy.array(self.test_predictions)
#    
#    prediction_history = PredictionHistory()
    
    tensorboard_logging = keras.callbacks.TensorBoard(log_dir='/storage/users/Muciaccia/burst/logs/{}/'.format(str(datetime.datetime.now())), # datetime.datetime.now().isoformat()
                                                      histogram_freq=0,
                                                      write_graph=False,
                                                      write_grads=False,
                                                      batch_size=32,
                                                      write_images=False,
                                                      embeddings_freq=0,
                                                      embeddings_layer_names=None,
                                                      embeddings_metadata=None)
    # TODO usage:
    # in the remote server: tensorboard --logdir /path/to/my/logs/
    # in the local machine: ssh -L 16006:127.0.0.1:6006 muciaccia@virgo-wn100
    # in my local browser: http://127.0.0.1:16006/
    
    # TODO magari fare un callback che salva dentro TensorBoard le immagini misclassificate quando la accuracy è sufficientemente alta (ad esempio > 0.95), in modo ca controllare in real-time come stanno andando le cose
    
    try:
        train_history = model.fit(train_images, train_classes,
    	    batch_size=minibatch_size,
    	    epochs=number_of_epochs,
    	    verbose=True,
    	    validation_data=(test_images, test_classes),
    	    #validation_split=0.5,
    	    shuffle=True, #'batch' # TODO hdf5 only supports batch-sized shuffling # train data shuffled at each epoch. validation data never shuffled
    	    callbacks=[iteration_history] # TODO tensorboard_logging # TODO prediction_history
    	    #callbacks=[early_stopping] # TODO mettere callback per la visualizzazione interattiva su TensorBoard
    	    # TODO far decrescere gradualmente il learning rate durante il curriculum learning
    	    # TODO far scrivere a intervalli regolari il numero di iterazioni (tipo ogni 100 iterazioni, che corrispondono a 6400 immagini, per poi poter fare il grafico del curriculum learning)
    	    )
    except KeyboardInterrupt: # TODO fare in modo che venga comunque salvata la history
        print('\n')
        print('manual early stopping!') # TODO automatizzare, magari tramite un callback
        break # exit from the for loop and then exit from the script
    
    # save the trained model
    model.save('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(signal_to_noise_ratio))
    print('model saved')
    # TODO oppure salvare solo i pesi, in modo da poter successivamente modificare l'entità del dropout
    
    # save the train history
    train_history = pandas.DataFrame(train_history.history)
    train_history.to_csv('/storage/users/Muciaccia/burst/models/training_history_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)
    
    # save the detailed train history
    # TODO metterlo direttamnete dentro .on_train_end() nel callback
    detailed_train_history = pandas.DataFrame({'loss': iteration_history.loss,
                                               'accuracy': iteration_history.accuracy})
    detailed_train_history.index.name = 'iteration'
    detailed_train_history.to_csv('/storage/users/Muciaccia/burst/models/detailed_training_history_SNR_{}.csv'.format(signal_to_noise_ratio), index=True)
    # TODO non si possono avere le quantità di test per la singola iterazione ma solo per la singola epoca
    
#    # save the prediction history
#    # TODO per fare il filmino con l'evoluzione degli istogrammi
#    # TODO e per controllare se è possibile monitorare l'overfit direttamente tramite l'istogramma degli outut del solo dataset di train
#    numpy.save('/storage/users/Muciaccia/burst/models/train_prediction_history_SNR_{}.npy'.format(signal_to_noise_ratio), prediction_history.train_predictions)
#    numpy.save('/storage/users/Muciaccia/burst/models/test_prediction_history_SNR_{}.npy'.format(signal_to_noise_ratio), prediction_history.test_predictions)
    
    ################################



