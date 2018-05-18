
# Copyright (C) 2018 Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


import pandas

import config

results_table = pandas.DataFrame()
for signal_to_noise_ratio in config.all_SNR:
    single_result = pandas.read_csv('/storage/users/Muciaccia/burst/models/results_SNR_{}.csv'.format(signal_to_noise_ratio), usecols=['SNR', 'efficiency (%)', 'false alarms (%)'])
    results_table = results_table.append(single_result)
    results_table.to_csv('/storage/users/Muciaccia/burst/models/results_table.csv', index=False)

