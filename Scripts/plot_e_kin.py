#!/usr/bin/python3

import os

import numpy as np
import matplotlib.pyplot as plt

from plotting import *
set_sns_standard()

residue_names = [['20 GLY', '21 ARG', '22 VAL', '23 TYR'], ['24 AZU'], ['25 PHE', '26 ASN', '27 HIS', '28 ILE']]
flat_residue_names = [res for sublist in residue_names for res in sublist]

E_kin_evolution = np.load(os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_residues_average.npy'))
time = np.load(os.path.expanduser('~/HiWi/WW/working_files/910K/time_scale.npy')) 

# create a canvas with one ax and plot
for indices in [list(range(4)), [4], list(range(5,9))]:

  fig = plt.figure()
  axe = fig.add_subplot(111)

  for i in indices:
    ekin = E_kin_evolution[i]
    axe.plot(time, ekin, label=flat_residue_names[i])

  axe.set_xscale('log')
  axe.set_xlabel('t / ps')
  axe.set_ylabel('$E_{kin} / kJ/mol$')
  axe.set_title('Time evolution of kinetic energies')
  axe.set_xlim(xmax=max(time))
  axe.grid()
  axe.legend(loc=0)
  remove_ticks(axe)
  
  fig.tight_layout()
  savename = os.path.expanduser('~/HiWi/WW/Plots/910K/E_kin_' + ''.join(flat_residue_names[indices[0]:indices[-1]]).replace(' ', '_'))
  fig.savefig(savename + '.png')
  fig.savefig(savename + '.pdf')

