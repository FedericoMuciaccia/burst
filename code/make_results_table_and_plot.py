
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

########################

import matplotlib
from matplotlib import pyplot
matplotlib.rcParams.update({'font.size': 23})

fig = pyplot.figure(figsize=[8,6])
ax1 = pyplot.subplot()
ax2 = ax1.twinx()

ax1_color = '#0033cc' # blue
ax1.set_title('trigger performances', y=1.05)
ax1.plot(results_table['SNR'], results_table['efficiency (%)'], color=ax1_color, linewidth=2, alpha=0.8)

ax1.set_xlabel('SNR', labelpad=10)
ax1.set_xticks(config.all_SNR)#, minor=True) # rotation=45, ha = "right"
ax1.xaxis.set_major_locator(pyplot.MultipleLocator(5))#10))
#ax1.xaxis.set_minor_locator(pyplot.MultipleLocator(5))

ax1.set_ylim(75-1.25, 100+1.25)#, auto=False)
ax1.yaxis.set_major_locator(pyplot.MultipleLocator(5))
ax1.set_ylabel('CNN efficiency (%)', color=ax1_color)
ax1.tick_params(axis='y', labelcolor=ax1_color)
#ax1.autoscale(enable=False)
#ax1.margins(0.05, 0.05, tight=False)

ax1.grid(linestyle='dashed', which='both')

ax2_color = '#b30000' # red
ax2.plot(results_table['SNR'], results_table['false alarms (%)'], color=ax2_color, linewidth=2, alpha=0.8) # linestyle='--'

ax2.set_ylim(0-0.25, 5+0.25)#, auto=False)
#ax2.set_yticks(range(5+1))
ax2.yaxis.set_major_locator(pyplot.MultipleLocator(1))
#ax2.set_ymargin(0.5)
#ax2.relim()
#ax2.margins(y=0.5, tight=False)
#ax2.autoscale(enable=True, tight=False)
#ax2.autoscale_view(...)
ax2.set_ylabel('CNN false alarms (%)', color=ax2_color, labelpad=20)
ax2.tick_params(axis='y', labelcolor=ax2_color)

fig.tight_layout()

pyplot.savefig('/storage/users/Muciaccia/burst/media/performances.jpg', bbox_inches='tight', transparent=True)
pyplot.savefig('/storage/users/Muciaccia/burst/media/performances.svg', bbox_inches='tight', transparent=True)
pyplot.show()
pyplot.close()


