
# Copyright (C) 2018  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy
import h5py
#import sklearn
import sklearn.utils
import sklearn.model_selection
import sklearn.preprocessing
import dask.array


def data_preparation(SNR):
    
    # dataset loading
    
    print('SNR:', SNR)
    
    level = 6 # TODO hardcoded
    # TODO fare diverse reti, una per livello, che collaborano nel prendere la decisione finale
    # TODO o magari anche una rete che ha in input le probabilità date dalle singole reti ai vari livelli e decide globalmente il da farsi
    
    #height = 2**level # 64 frequency divisions
    #width = 256 # time bins
    #channels = 3 # number of detectors
    
    signal_file_path = '/storage/users/Muciaccia/burst/data/new_data/SNR{}.hdf5'.format(SNR)
    signal_images = h5py.File(signal_file_path)['spectro']
    signal_number_of_samples, height, width, channels = signal_images.shape
    print('signal images:', signal_number_of_samples)
    
    noise_file_path = '/storage/users/Muciaccia/burst/data/new_data/Noise.hdf5'
    noise_images = h5py.File(noise_file_path)['spectro']
    noise_number_of_samples, height, width, channels = noise_images.shape
    print('noise images:', noise_number_of_samples)
    
    #########################
    
    # dataset merging
    
    # the two classes should be equipopulated # TODO imporlo in fase di costruzione del dataset
    number_of_samples = numpy.min([signal_number_of_samples, noise_number_of_samples, 50000]) # TODO con 100000 dà MemoryError # TODO in futuro usare direttamente la input pipeline di TensorFlow
    
    # avoid the last minibacth at the end of every epoch to be smaller than all the others
    minibatch_size = 64 # TODO hardcoded
    number_of_samples = minibatch_size * numpy.floor_divide(number_of_samples, minibatch_size)
    print('number_of_samples:', number_of_samples)
    
    signal_images = h5py.File(signal_file_path)['spectro'][slice(number_of_samples)] # TODO lentissimo
    noise_random_index = numpy.random.randint(noise_number_of_samples - number_of_samples) # TODO sistemare meglio facendo magari direttamente un veloce shuffle del noise direttamente sugli indici del minibatch
    noise_images = h5py.File(noise_file_path)['spectro'][slice(noise_random_index, noise_random_index + number_of_samples)] # TODO lentissimo
    
    signal_classes = numpy.ones(number_of_samples)
    noise_classes = numpy.zeros(number_of_samples)
    
    images = numpy.concatenate([signal_images, noise_images]) # TODO lentissimo # TODO vedere se la concatenazione comporta un in utile spreco del doppio della memoria
    classes = numpy.concatenate([signal_classes, noise_classes])
    # TODO vedere nuova pipeline standard di input per TensorFlow
    
    #########################
    
    # data shuffling
    
    # TODO forse il set di test/validazione non andrebbe mischiato, in modo da poter controllare gli indici delle immagini misclassificate
    images, classes = sklearn.utils.shuffle(images, classes) # TODO lento
    
    #########################
    
    # train-test splitting
    
    splitting_ratio = 0.5 # we want the same statistical fluctuations when we compare the train and the test datasets
    
    train_images, test_images, train_classes, test_classes = sklearn.model_selection.train_test_split(images, classes, test_size= splitting_ratio, shuffle=True) # TODO lento # TODO attenzione al nuovo shuffling e ai nuovi indici
    # TODO vedere nuova input pipeline di TensorFlow
    
    #########################
    
    # categorical (one-hot) encoding
    
    number_of_classes = 2 # TODO hardcoded # TODO dopo farne 4, coi glitch
    to_categorical = sklearn.preprocessing.OneHotEncoder(n_values=number_of_classes, sparse=False, dtype=numpy.float32)
    train_classes = to_categorical.fit_transform(train_classes.reshape(-1,1))
    test_classes = to_categorical.fit_transform(test_classes.reshape(-1,1))
    # TODO farlo in maniera più chiara ed elegante, magari direttamente con TensorFlow
    
    # TODO vedere se adesso TensorFlow ha delle pipeline di input più ottimizzate
    
    #########################
    
    # saving to disk
    
    train_images = dask.array.from_array(train_images, chunks=(minibatch_size, height, width, channels))
    train_classes = dask.array.from_array(train_classes, chunks=(minibatch_size, number_of_classes))
    
    test_images = dask.array.from_array(test_images, chunks=(minibatch_size, height, width, channels))
    test_classes = dask.array.from_array(test_classes, chunks=(minibatch_size, number_of_classes))
    
    # saving frequency and time indices is useless here
    # but if you want you can add (optional):
    # a time array of 256 float64 GPS/ISO times
    # a frequency array of 256 float64 frequencies in Hz
    # (just to have something to plot on the spectrogram axes)
    
    file_path = '/storage/users/Muciaccia/burst/data/preprocessed/SNR_{}.hdf5'.format(SNR)
    
    # TODO aggiungere attributi attrs al dataset (class, SNR)
    # TODO attenzione che bisogna prima cancellare i precedenti file, perché sembra che la libreria tenti di aggiornare quelli
    # TODO mettere un warning esplicito se la cartella di destinazione non appare vuota
    # TODO magari usare direttamente netCDF4 che mi sembra più flessibile, oppure usare direttamente la input pipeline di TensorFlow
    dask.array.to_hdf5(file_path,
                       {'/train_images': train_images,
                        '/train_classes': train_classes,
                        '/test_images': test_images,
                        '/test_classes': test_classes},
                       #compression='gzip', # good compression, low speed
                       #compression='lzf', # poorer compression, faster speed
                       chunks=True) # auto-chunking # TODO CHECK 64
                       #shuffle=True # TODO
    
    print('saved', file_path)

#########################

if __name__ == '__main__':
    
    signal_to_noise_ratio = [40, 35, 30, 25, 20, 15, 12, 10, 8] # TODO hardcoded
    # SNR images
    # 40 11522
    # 35 17224
    # 20 22779
    # 25 45122
    # 20 109757
    # 15 87540
    # 12 143899
    # 10 68079
    # 8 25346
    # noise 285209
    
    for SNR in signal_to_noise_ratio:
        data_preparation(SNR)

