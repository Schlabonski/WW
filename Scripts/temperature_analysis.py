#!/usr/bin/python3

import os
import numpy as np
import subprocess as sub

basename = 'ww_24_13_heating_'
traj_path = '/scratch/basti/heating/'
index_path = traj_path + '../ww_24_13.ndx'

for i in range(100):
  trr = traj_path + basename + str(i) + '.trr'
  tpr = traj_path + basename + str(i) + '.tpr'
  T_out = os.path.expanduser('~/HiWi/WW/working_files/temperatures/{0}temp_{1}.xvg'.format(basename, str(i)))

  commands = ['g_traj', '-f', trr, '-s', tpr, '-n', index_path, '-ot', T_out]

  Process = sub.Popen(commands, stdin=sub.PIPE, stdout=sub.PIPE)
  Process.communicate('19 \n')
  Process.wait()
