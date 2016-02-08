#!/usr/bin/python3

import os

import trajectory_class as traj 

traj_path = '/u2/WW/NEQ/run_910K/'
traj_name = 'ww_24_13_NEQ_'
traj_name_trr = 'ww_24_13_NEQ_prot_'


for i in  range(1, 100):
  Traj = traj.Trajectory(traj_path=traj_path, basename=traj_name+str(i), basename_trr=traj_name_trr+str(i))
  Traj.set_working_dir(os.path.expanduser('~/HiWi/WW/working_files/910K'))

  Traj.analyse_velocities()
  Traj.atom_masses('/u2/WW/ww_24_13.top')
  Traj.velocities_to_npy()
  Traj.kinetic_energies()



# do ensemble statistics
Ensemble = traj.Statistical_ensemble(path=traj_path, basename=traj_name, iterator=range(1,100), basename_trr=traj_name_trr)

residue_names = ['15 MET', '16 SER', '17 ARG', '18 SER', '20 GLY', '21 ARG', '22 VAL', '23 TYR', '25 PHE', '26 ASN', '27 HIS', '28 ILE', '24 AZU']
atom_indices = [list(range(157, 174)), list(range(174, 185)), list(range(184,209)), list(range(209,220)), list(range(231,238)), list(range(238,262)), list(range(262,278)), list(range(278,299)), list(range(325,345)), list(range(345, 359)), list(range(359,376)), list(range(376, 395)), list(range(278,299))]
residues = []
for i in range(len(residue_names)):
  residues.append([residue_names[i]]+atom_indices[i])

Ensemble.E_kin_evolution(residues)
Ensemble.plot_E_kin(residue_names[0:4])
Ensemble.plot_E_kin(residue_names[4:8])
Ensemble.plot_E_kin(residue_names[8:])
