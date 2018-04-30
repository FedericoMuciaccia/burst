
# Copyright (C) 2017 Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


#import keras
import h5py
import numpy
import pandas
import glob

import matplotlib
from matplotlib import pyplot
matplotlib.rcParams.update({'font.size': 23}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

# #vari RGB ad SNR ricostruito 40 per mostrare quanto è utile ed espressivo l'RGB
# #abbandono della likelihood
# falsi positivi e problemi di ghost e pixeloni
# #SNR di network e veto su due detector singolarmente ad SNR >4, perché il terzo può essere spento o non vedere nulla a causa del pattern d'antenna
# purezza di cWB assunta 100% tramite veto sul timing (per eliminare i falsi positivi che c'erano prima)
# #plot delle forme d'onda nel tempo
# esito della confusion matrix
# esito delle metriche ai vari SNR
# #futura rete composita e cooperativa
# grafico del training completo
# #futuro grafico della regione di cielo

levels = numpy.array([5,6,7,8])
level = 6

SNR = numpy.array([40, 35, 30, 25, 20, 15, 10])
signal_to_noise_ratio = 40 # 40 35 30 25 20 15 10

#signal_images = h5py.File('/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/level_{}.hdf5'.format(signal_to_noise_ratio, level))['spectro']
signal_images = h5py.File('/storage/users/Muciaccia/burst/data/new_data/SNR40.hdf5')['spectro']

signal_likelihoods = h5py.File('/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/likelihood.hdf5'.format(signal_to_noise_ratio))['spectro']

noise_images = h5py.File('/storage/users/Muciaccia/burst/data/new_data/SNR_40_OLD/noise_level_{}.hdf5'.format(level))['spectro']

noise_likelihoods = h5py.File('/storage/users/Muciaccia/burst/data/big_set_gaussian_white_noise/likelihood.hdf5'.format(signal_to_noise_ratio))['spectro']

waveform_paths = glob.glob('/storage/users/Muciaccia/burst/data/waveforms Pablo Cerdá-Durán/g-modes/*.h5')

#########################

for index in range(16): # TODO separare tutti questi plot in blocchi differenti ed eseguibili singolarmente
    signal_image = signal_images[index]
    
    pyplot.figure(figsize=[12,10])
    # red = Hanford, green = Livingston, blue = Virgo
    pyplot.title('RGB time-frequency plane')
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(signal_image, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/SNR_{}_level_{}_index_{}.jpg'.format(signal_to_noise_ratio, level, index), bbox_inches='tight')
    pyplot.close()
    
    signal_likelihood = signal_likelihoods[index]
    
    pyplot.figure(figsize=[10,10])
    pyplot.title('SNR {} multilevel likelihood'.format(signal_to_noise_ratio))
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(signal_likelihood, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/SNR_{}_likelihood_index_{}.jpg'.format(signal_to_noise_ratio, index), bbox_inches='tight')
    pyplot.close()
    
    noise_image = noise_images[index]
    
    pyplot.figure(figsize=[12,10])
    # red = Hanford, green = Livingston, blue = Virgo
    pyplot.title('RGB time-frequency plane')
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(noise_image, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/noise_level_{}_index_{}.jpg'.format(level, index), bbox_inches='tight')
    pyplot.close()
    
    noise_likelihood = noise_likelihoods[index]
    
    pyplot.figure(figsize=[10,10])
    pyplot.title('multilevel likelihood')
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(noise_likelihood, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/noise_likelihood_index_{}.jpg'.format(index), bbox_inches='tight')
    pyplot.close()
    
    waveform = h5py.File(waveform_paths[index])
    strain = waveform['strain']
    time = waveform['time']
    
    pyplot.figure(figsize=[9,6])
    pyplot.title('waveform')
    pyplot.plot(time, strain)
    pyplot.xlabel('time') # TODO vedere scala temporale
    pyplot.ylabel('strain') # TODO vedere scala di strain
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/waveform_index_{}.jpg'.format(index), bbox_inches='tight')
    pyplot.close()
    # TODO spiegare che non c'è corrispondenza tra gli indici delle waveform e gli indici degli scalogrammi

#######################

# training history

start_acc = 0.5
start_loss = numpy.log(2)

starting_point = pandas.DataFrame({'acc': start_acc,
                                   'loss': start_loss,
                                   'val_acc': start_acc,
                                   'val_loss': start_loss}, index=[0])

total_history = [starting_point]
epochs = []
for i in SNR:
    partial_history = pandas.read_csv('/storage/users/Muciaccia/burst/models/training_history_SNR_{}.csv'.format(i))
    epochs.append(len(partial_history))
    total_history.append(partial_history)
total_history = pandas.concat(total_history, ignore_index=True)
epochs = numpy.array(epochs)
cumulative_epochs = epochs.cumsum()
cumulative_epochs = cumulative_epochs

# TODO fare questo grafico in iterations e non in epochs, perché queste ultime sono deformate dalla grandezza del dataset
fig, [ax1, ax2] = pyplot.subplots(2, sharex=True, figsize=(10,6))
fig.suptitle('model performances', size=25) # 12 # TODO fontsize=25
# train 1 - accuracy
ax1.plot(1-total_history.acc, label='train', color='blue')
# test 1 - accuracy
ax1.plot(1-total_history.val_acc, label='test', color='green')
ax1.set_ylabel('classification error') # r'$1-$accuracy' = 'error'
ax1.legend(loc='lower right', frameon=False, fontsize=20)
ax1.set_xlim((0,cumulative_epochs[-1]))
ax1.set_ylim((1e-5,1e-0))
ax1.set_yscale('log')
ax1.vlines(x=cumulative_epochs, ymin=1e-5 ,ymax=1e-0, color='gray', linestyle='dashed') # OR 'dotted'
# train loss
ax2.plot(total_history.loss, label='train', color='blue')
# test loss
ax2.plot(total_history.val_loss, label='test', color='green')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(loc='lower right', frameon=False, fontsize=20)
ax2.set_ylim((1e-5,1e-0))
ax2.set_yscale('log')
ax2.vlines(x=cumulative_epochs, ymin=1e-5 ,ymax=1e-0, color='gray', linestyle='dashed') # OR 'dotted'
#pyplot.show()
fig.savefig('/storage/users/Muciaccia/burst/media/training_history.jpg', bbox_inches='tight')
fig.savefig('/storage/users/Muciaccia/burst/media/training_history.svg', bbox_inches='tight') # TODO facendo così è possibile salvare i grafici due volte in due formati diversi!
pyplot.close()

############################

# curriculum learning

for i in SNR:
    signal_images = h5py.File('/storage/users/Muciaccia/burst/data/new_data/SNR{}.hdf5'.format(i))['spectro']
    
    index = 0 # numpy.random.randint(100)
    signal_image = signal_images[index]
    
    pyplot.figure(figsize=[12,10])
    # red = Hanford, green = Livingston, blue = Virgo
    pyplot.title('SNR {}'.format(i)) # 'RGB time-frequency plane'
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(signal_image, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/SNR_{}.jpg'.format(i), bbox_inches='tight')
    pyplot.close()

########################

# level choice

for l in levels:
    signal_images = h5py.File('/storage/users/Muciaccia/burst/data/g_modes/SNR_50/level_{}.hdf5'.format(l))['spectro']
    # TODO probabilmente questo dataset contiene errori di allineamento delle immagini, perché molte sembrano in diagonale
    
    index = 1 # numpy.random.randint(100)
    signal_image = signal_images[index]
    
    pyplot.figure(figsize=[12,10])
    # red = Hanford, green = Livingston, blue = Virgo
    pyplot.title('RGB time-frequency plane')
    pyplot.xlabel('time [bin]') # TODO vedere scala temporale
    pyplot.ylabel('frequency [bin]') # TODO vedere scala di frequenze
    pyplot.imshow(signal_image, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/level_{}.jpg'.format(l), bbox_inches='tight')
    pyplot.close()

#######################

# ROC curve

minimum_SNR = SNR.min()

ROC_curve = pandas.read_csv('/storage/users/Muciaccia/burst/models/ROC_curve_SNR_{}.csv'.format(minimum_SNR))

default_working_point = ROC_curve[ROC_curve.threshold == 0.5]

pyplot.figure(figsize=[8,8])
pyplot.title('ROC curve')
# TODO pyplot.tight_layut()
pyplot.scatter(ROC_curve.false_alarm, ROC_curve.efficiency)
pyplot.scatter(default_working_point.false_alarm, default_working_point.efficiency, color='#ff5500', s=150)
pyplot.xlabel('false alarm (%)')
pyplot.ylabel('efficiency (%)')
#pyplot.show()
pyplot.savefig('/storage/users/Muciaccia/burst/media/ROC_curve_SNR_{}.jpg'.format(minimum_SNR), bbox_inches='tight')
pyplot.savefig('/storage/users/Muciaccia/burst/media/ROC_curve_SNR_{}.svg'.format(minimum_SNR), bbox_inches='tight')
pyplot.close()

# TODO mettere tutto su un editor che supporti i blocchi di codice e mettere ogni singolo grafico dentro un diverso blocco, in modo da poterli eseguire e rieseguire separatamente

######################

#signals:
#niente pixeloni
#niente ghost grossi?
#problemi con segnali ai bordi dell'immagine?
#problemi con i segnali verdi?
#curva di distribuzione degli SNR iniettati e ricostruiti?

