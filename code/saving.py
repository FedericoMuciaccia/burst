
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


"""
this little python3 script shows how to properly save an hdf5 dataset
"""

import numpy
import dask.array # creating an hdf5 file is easier with dask than with plain h5py

# the shape of every dataset should be [number_of_images, height, width]

number_of_images = 1024 # let's start creating a small dataset
height = 256
width = 256

# generate 1024 dummy spectrograms 256x256 pixels for every detector (float32 values between 0 and 1)
H = numpy.random.rand(number_of_images, height, width).astype(numpy.float32)
L = numpy.random.rand(number_of_images, height, width).astype(numpy.float32)
V = numpy.random.rand(number_of_images, height, width).astype(numpy.float32)

# generate 1024 dummy likelihood images 256x256 pixels (float32 values between 0 and 1)
likelihood = numpy.random.rand(number_of_images, height, width).astype(numpy.float32)

# from now on use your real data instead ;)

# pass the data to Dask, for out-of-core computing
# dask arrays are transparently handled by numpy
minibatch_size = 64
likelihood = dask.array.from_array(likelihood, chunks=(minibatch_size, height, width))
H = dask.array.from_array(H, chunks=(minibatch_size, height, width))
L = dask.array.from_array(L, chunks=(minibatch_size, height, width))
V = dask.array.from_array(V, chunks=(minibatch_size, height, width))

label = 'SASI' # the name of our class
signal_noise_ratio = 50

file_name = '{}_SNR_{}.hdf5'.format(label, signal_noise_ratio)

# saving frequency and time indices is useless here
# but if you want you can add (optional):
# a time array of 256 float64 GPS/ISO times
# a frequency array of 256 float64 frequencies in Hz
# (just to have something to plot on the spectrogram axes)

# TODO aggiungere attributi attrs al dataset (class, SNR)

dask.array.to_hdf5(file_name,
                   {'/H':H,
                    '/L':L,
                    '/V':V,
                    '/likelihood': likelihood},
                   #compression='gzip', # good compression, low speed
                   #compression='lzf', # poorer compression, faster speed
                   chunks=True)

################################################

# now try to re-read the saved file

import h5py
from matplotlib import pyplot

dataset = h5py.File(file_name)

def view_image(image):
    pyplot.figure(figsize=[10,10])
    pyplot.imshow(image, interpolation='none', origin="lower")
    pyplot.show()

# plot the spectrogram with index=100 from the Hanford detector
view_image(dataset['H'][100])



