
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
from matplotlib import pyplot

# #vari RGB ad SNR ricostruito 40 per mostrare quanto è utile ed espressivo l'RGB
# buone immagini al decrescere dell'SNR fino ad arrivare a 10, dove non si dovrebbe vedere quasi nulla
# immagine di white noise
# immagini ai vari livelli e scelta del livello 6
# #abbandono della likelihood
# architettura della rete
# summary del modello e numero di parametri e altre variabili scelte come Adam, inizializzazione, batch_size, ecc
# falsi positivi e problemi di ghost e pixeloni
# #SNR di network e veto sul singolo superiore a 4 # TODO sarebbe più giusto mettere il veto su due detector singolarmente ad SNR >4, perché il terzo può essere spento o non vedere nulla a causa del pattern d'antenna
# purezza di cWB assunta 100% tramite veto sul timing (per eliminare i falsi positivi che c'erano prima)
# #plot delle forme d'onda nel tempo
# esito della confusion matrix
# esito delle metriche ai vari SNR
# #futuro blind test
# #futura rete composita e cooperativa
# grafico del training completo
# #futuro grafico della regione di cielo

level = 6

signal_to_noise_ratio = 40 # 40 35 30 25 20 15 10

signal_images = h5py.File('/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/level_{}.hdf5'.format(signal_to_noise_ratio, level))['spectro']

likelihoods = h5py.File('/storage/users/Muciaccia/burst/data/INCOMPLETE_g_modes_cut_SNR_4/SNR_{}/likelihood.hdf5'.format(signal_to_noise_ratio))['spectro']

waveform_paths = glob.glob('./waveforms Pablo Cerdá-Durán/g-modes/*.h5')

#0 bianco
#1 giallo
#2 rosa con ghost verde?
#3 bianco-verde
#4 blu-bianco
#5 7 verde-giallo
#8 magenta
#9 giallo con macchia nera?
#11 giallo-verde
#13 ciano
#14 bianco-ciano
for index in range(16):
    signal_image = signal_images[index]
    
    pyplot.figure(figsize=[15,10])
    pyplot.title('''
                 RGB time-frequency plane
                 red = Hanford, green = Livingston, blue = Virgo
                 SNR {}, level {}
                 g-mode supernova burst
                 '''.format(signal_to_noise_ratio, level))
    pyplot.xlabel('time') # TODO vedere scala temporale
    pyplot.ylabel('frequency') # TODO vedere scala di frequenze
    pyplot.imshow(signal_image, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/SNR_{}_level_{}_index_{}.jpg'.format(signal_to_noise_ratio, level, index), dpi=300)
    pyplot.close()
    
    likelihood = likelihoods[index]
    
    pyplot.figure(figsize=[15,10])
    pyplot.title('''
                 SNR {} multilevel likelihood
                 g-mode supernova burst
                 '''.format(signal_to_noise_ratio))
    pyplot.xlabel('time') # TODO vedere scala temporale
    pyplot.ylabel('frequency') # TODO vedere scala di frequenze
    pyplot.imshow(likelihood, interpolation='none', origin="lower")
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/burst/media/SNR_{}_likelihood_index_{}.jpg'.format(signal_to_noise_ratio, index), dpi=300)
    pyplot.close()

    waveform = h5py.File(waveform_paths[index])
    strain = waveform['strain']
    time = waveform['time']
    
    pyplot.figure(figsize=[9,6])
    pyplot.title('g-mode supernova burst waveform')
    pyplot.plot(time, strain)
    pyplot.xlabel('time') # TODO vedere scala temporale
    pyplot.ylabel('strain') # TODO vedere scala di strain
    pyplot.show()
    #pyplot.savefig('/storage/users/Muciaccia/burst/media/waveform_index_{}.jpg'.format(index), dpi=300)
    pyplot.close()
    # TODO spiegare che non c'è corrispondenza tra gli indici delle waveform e gli indici degli scalogrammi

