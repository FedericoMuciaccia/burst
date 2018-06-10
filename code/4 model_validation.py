
# Copyright (C) 2018 Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras
import h5py # TODO vedere se persiste la costrizione di importarlo dopo keras
import numpy
import sklearn.metrics
import pandas

import matplotlib
from matplotlib import pyplot
matplotlib.rcParams.update({'font.size': 25}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

import config

level = config.cWB_level
all_SNR = numpy.array(config.all_SNR)

minimum_SNR = all_SNR.min()

for signal_to_noise_ratio in all_SNR:
    print('SNR:', signal_to_noise_ratio)

    #signal_to_noise_ratio = 40
    
    #########################
    
    # loading test data
    
    dataset = h5py.File('/storage/users/Muciaccia/burst/data/preprocessed/SNR_{}.hdf5'.format(signal_to_noise_ratio))
    # TODO usare dask.array?
    
    # TODO greedy VS lazy?
    test_images = dataset['test_images'].value
    test_classes = dataset['test_classes'].value
    # TODO vedere la nuova input pipeline di TensorFlow
    
    #########################
    
    # model predictions
    
    model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(minimum_SNR)) #signal_to_noise_ratio # TODO inutilmente ripetuto
    
    predictions = model.predict(test_images, batch_size=128, verbose=1) # the minibatch size doesn't seem to influence the prediction time
    predicted_signal_probabilities = predictions[:,1]
    true_classes = test_classes[:,1]
    
    threshold = 0.5 # TODO fine tuning ed istogramma
    predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)
    
    # TODO salvare i valori di predicted_signal_probabilities e predicted_classes per poter fare i grafici velocemente in un secondo momento
    
    ########################
    
    # visualize some misclassified images
    
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
    # TODO non capisco perché i segnali misclassificati tendono ad essere quelli colorati di verde (c'é una qualche asimmetria tra i detector che data la diversa sensibilità dei rivelatori ne favorisce uno a parità di SNR di network?)
    
    #########################
    
    # compute the metrics
    
    confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
    
    def compute_the_metrics(confusion_matrix):
        [[true_negatives,false_positives],[false_negatives,true_positives]] = [[predicted_0_true_0,predicted_1_true_0],[predicted_0_true_1,predicted_1_true_1]] = confusion_matrix
        purity = true_positives/(true_positives + false_positives) # precision: how many selected events are signals?
        efficiency = true_positives/(true_positives + false_negatives) # recall: how many signal events are selected? # sensitivity
        accuracy = (true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives)
        # normalizations
        all_real_signals = true_positives + false_negatives # "signal events"
        all_real_noises = true_negatives + false_positives
        all_predicted_as_signals = true_positives + false_positives # "selected events"
        all_predicted_as_noise = true_negatives + false_negatives
        all_validation_samples = true_positives + false_positives + true_negatives + false_negatives
        
        metrics = {#'SNR':signal_to_noise_ratio,
                   #'level':level,
                   'all validation samples':all_validation_samples,
                   'misclassified images':false_positives+false_negatives,
                   'false negatives':false_negatives,
                   'false positives':false_positives,
                   'rejected noise (%)':100*true_negatives/all_real_noises, # specificity
                   'false alarms (%)':100*false_positives/all_predicted_as_signals, #all_real_noises, # TODO capire meglio!!! (dovrebbe essere 1 - purity) # "false alarm rate"
                   'missed signals (%)':100*false_negatives/all_real_signals, # TODO il false dismissal è 1-efficiency ?
                   'selected signals (%)':100*true_positives/all_real_signals,
                   'purity (%)':100*purity,
                   'efficiency (%)':100*efficiency,
                   'accuracy (%)':100*accuracy}
        return metrics
    
    metrics = compute_the_metrics(confusion_matrix)
    metrics.update({'SNR':signal_to_noise_ratio, 'level':level})
    results = pandas.DataFrame(metrics, index=[signal_to_noise_ratio])
    results.to_csv('/storage/users/Muciaccia/burst/models/results_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)
    # TODO poi concatenarli con pandas.concat()
    # TODO magari levare le percentuali e rimettere le quantità normalizzate
    
    # NOTA: ai fini della scoperta con 5 sigma di confidenza bisogna guardare il valore di purezza (o di false alarm)
    
    # a mio avviso si sviluppa un leggero overfitting ad SNR 15 ed SNR 10
    
    # TODO provare a diminiure gradualmente sia il dropout che il learning rate
    
    ######################
    
    # plot the output histogram
    
    # TODO sistemare le dimensioni del plot
    
    fig_predictions = pyplot.figure(figsize=[12,8]) # 9, 6
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
    ax1.set_title('classifier output (SNR {})'.format(signal_to_noise_ratio)) # OR 'model output'
    ax1.set_ylabel('count') # OR density
    ax1.set_xlabel('predicted probability to be in the "signal" class') # OR 'class prediction'
    tick_spacing = 0.1
    ax1.set_yscale('log')
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacing))
    #ax1.legend(loc='best')
    pyplot.axvline(x=0.5, # TODO best_threshold
    	           color='grey', 
    	           linestyle='dotted', 
    	           alpha=0.8)
    ax1.legend(loc='upper center')#, frameon=False)
    #pyplot.show()
    fig_predictions.savefig('/storage/users/Muciaccia/burst/media/classifier_output_SNR_{}.jpg'.format(signal_to_noise_ratio), bbox_inches='tight')
    fig_predictions.savefig('/storage/users/Muciaccia/burst/media/classifier_output_SNR_{}.svg'.format(signal_to_noise_ratio), bbox_inches='tight')
    pyplot.show()
    pyplot.close()
    
    #############################
    
    # ROC curve
    
    
    # def compute_ROC_curve(predicted_signal_probabilities, true_classes):
    
    thresholds = numpy.linspace(start=0 ,stop=1 ,num=101) # 100 bins
    thresholds = thresholds[1:-1] # to avoid problems at the borders
    
    efficiency_list = []
    false_alarm_list = []
    
    for threshold in thresholds:
        predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)
        
        confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
        [[true_negatives,false_positives],[false_negatives,true_positives]] = [[predicted_0_true_0,predicted_1_true_0],[predicted_0_true_1,predicted_1_true_1]] = confusion_matrix
        
        efficiency = true_positives/(true_positives + false_negatives)
        false_alarm = false_positives/(true_positives + false_positives)
        #purity = true_positives/all_predicted_as_signals
        
        efficiency_list.append(efficiency)
        false_alarm_list.append(false_alarm)
    
    efficiency_list = numpy.array(efficiency_list)
    false_alarm_list = numpy.array(false_alarm_list)
    
    ROC_curve = {#'SNR':signal_to_noise_ratio,
                 #'level':level,
                 #'all validation samples':all_validation_samples,
                 'threshold':thresholds,
                 'false_alarm':100*false_alarm_list,
                 #'purity (%)':100*purity,
                 'efficiency':100*efficiency_list}
    
    ROC_curve = pandas.DataFrame(ROC_curve, index=thresholds)
    
    
    ROC_curve.to_csv('/storage/users/Muciaccia/burst/models/ROC_curve_SNR_{}.csv'.format(signal_to_noise_ratio), index=False)
    # TODO magari levare le percentuali e rimettere le quantità normalizzate
    
    # TODO magari far fare direttamente qui il plot
    
    ##########################


