
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


import dask.dataframe
import dask.array

levels = [5,6,7,8]
for level in levels:
    height=2**level
    whidth=256
    
    data_path='./LEV{}/Event_*.txt'.format(level)
    
    dataframe = dask.dataframe.read_csv(urlpath=data_path, delim_whitespace=True, header=None, usecols=[2,3,4], names=['H','L','V'], dtype='float32')
    
    H = dask.array.asarray(dataframe.H).reshape(-1,whidth,height).transpose([0,2,1])
    L = dask.array.asarray(dataframe.L).reshape(-1,whidth,height).transpose([0,2,1])
    V = dask.array.asarray(dataframe.V).reshape(-1,whidth,height).transpose([0,2,1])
    
    RGB_spectrogram = dask.array.stack([H,L,V], axis=-1)
    
    file_name = './level_{}.hdf5'.format(level)
    
    dask.array.to_hdf5(file_name,
                       {'/RGB_spectrogram':RGB_spectrogram},
                       #compression='gzip', # good compression, low speed
                       #compression='lzf', # poorer compression, faster speed
                       chunks=True)

################################################

# now try to re-read the last saved file

import h5py
from matplotlib import pyplot

dataset = h5py.File(file_name)

def view_image(image):
    pyplot.imshow(image, interpolation='none', origin="lower")
    pyplot.show()
    #pyplot.savefig('example.jpg', dpi=300)

# plot the first RGB spectrogram (index=0, all the detectors superimposed)
view_image(dataset['RGB_spectrogram'][0])





