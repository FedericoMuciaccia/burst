
# Copyright (C) 2018 Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify  it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras
import h5py
import numpy
import dask.array

file_path = '/storage/users/Muciaccia/burst/data/new_data/BlindSet.hdf5'

images = h5py.File(file_path)['spectro']
number_of_images, time, frequency, detector = images.shape
images = dask.array.from_array(images, chunks=[64, time, frequency, detector])
# TODO magari si può leggere il file hdf5 direttamente con Dask

minimum_trained_signal_to_noise_ratio = 10 # TODO hardcoded
print('trained SNR:', minimum_trained_signal_to_noise_ratio)

model = keras.models.load_model('/storage/users/Muciaccia/burst/models/trained_model_SNR_{}.hdf5'.format(minimum_trained_signal_to_noise_ratio))

predictions = model.predict(images, batch_size=128, verbose=1)

predicted_signal_probabilities = predictions[:,1]

numpy.savetxt('/storage/users/Muciaccia/burst/data/blind_validation/predicted_signal_probabilities_SNR_{}.txt'.format(minimum_trained_signal_to_noise_ratio), predicted_signal_probabilities, fmt='%f')

threshold = 0.5 # TODO fine tuning ed istogramma (scegliere il punto di lavoro sulla curva ROC)
predicted_classes = numpy.greater(predicted_signal_probabilities, threshold)

numpy.savetxt('/storage/users/Muciaccia/burst/data/blind_validation/predicted_classes_SNR_{}.txt'.format(minimum_trained_signal_to_noise_ratio), predicted_classes, fmt='%i')





