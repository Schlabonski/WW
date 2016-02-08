import MDAnalysis as md
import numpy as np
import os

traj_path = '/u2/WW/NEQ/run_910K/'
traj_name = 'ww_24_13_NEQ_'
traj_name_trr = 'ww_24_13_NEQ_prot_'

#topology = '/u2/WW/ww_24_13.top'

E_kin_all_residues = None
time = None

for i in  range(1, 100):
  trajectory = traj_path + traj_name_trr + str(i) + '.trr'
  topology   = traj_path + traj_name + str(i) + '.gro'
  un         = md.Universe(topology, trajectory)
  
  residues_of_interest   = [res for res in un.residues if res.id in range(20, 29)]
  residues_total_masses  = [] 

# initiate the array that will contain all time rows of kinetic energy
  if E_kin_all_residues == None:
    E_kin_all_residues = np.zeros((len(residues_of_interest), len(un.trajectory)))
    time               = np.zeros(len(un.trajectory))

  E_kin_all_residues_this_trajectory =  np.zeros((len(residues_of_interest), len(un.trajectory)))

# get the masses of all atoms in each of the residues of interest (array)
  for n in range(len(residues_of_interest)):
    residues_total_masses.append(residues_of_interest[n].atoms.masses())
 
# for each frame in the trajectory we calculate the total kinetic energy of 
# every residue of interest and put in in the timerow of kinetic energies
  time_steps = len(un.trajectory)
  print('Trajectory %s' % trajectory)
  for i_ts, ts in enumerate(un.trajectory):
    for i_res, res in enumerate(residues_of_interest):
      velocities   = res.atoms.get_velocities()
      v_square_sum = np.sum(velocities**2, axis=1)
      E_kin        = v_square_sum * residues_total_masses[residues_of_interest.index(res)]
      E_kin_tot    = np.sum(E_kin)
      
      E_kin_all_residues_this_trajectory[i_res, i_ts] += E_kin_tot
    
    time[i_ts] = un.trajectory.time

    if i_ts % 100.0 == 0.0:
      print '>> %3.2f \r' % (float(i_ts)/time_steps*100)
  print('Storing kinetic energy in ' + os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_residues_'+str(i)+'.npy')) 
  np.save(os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_residues_'+str(i)+'.npy'), E_kin_all_residues_this_trajectory) 
  E_kin_all_residues += E_kin_all_residues_this_trajectory

# average over the whole ensemble of trajectories
for line in E_kin_all_residues_this_trajectory:
  line /= len(range(1,100))

np.save(os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_residues_average.npy'), E_kin_all_residues_this_trajectory)
np.save(os.path.expanduser('~/HiWi/WW/working_files/910K/time_scale.npy'), time)
