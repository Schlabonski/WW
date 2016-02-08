#!/usr/bin/python3

import os
import numpy as np
import matplotlib.pyplot as plt
from plotting import *

set_sns_standard()

fig = plt.figure()
axe = fig.add_subplot(111)

basename = 'ww_24_13_heating_'

for i in [5,15,25,35,65,85]:
  T_out = os.path.expanduser('~/HiWi/WW/working_files/temperatures/{0}temp_{1}.xvg'.format(basename, str(i)))
  data = np.loadtxt(T_out, skiprows=19)
  t = data[:,0]
  temp = data[:,1]

# plotting begins here

  axe.plot(t, temp, label='Run {0}'.format(i))

remove_ticks(axe)
axe.legend(loc=0)
axe.set_xlabel('t/ps')
axe.set_ylabel('T/K')
axe.set_title('Temperature of AZU 24 during heating')
axe.grid()

fig.tight_layout()
fig.savefig(os.path.expanduser('~/HiWi/WW/Plots/azu_heating.png'))

plt.show()
