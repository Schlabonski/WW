#!~/anaconda2/bin/python

import os
import numpy as np

from master_equation_simulation import MEQ_Simulation as msim

working_dir = os.path.expanduser('~/HiWi/WW/working_files/910K/')   

root_path = '/u2/WW/NEQ/run_910K/'
top_name = 'ww_24_13_NEQ_1.gro'
trr_name = 'ww_24_13_NEQ_prot_1.trr'
meq = msim(root_path + top_name, root_path + trr_name)

meq.read_distance_variances('/u3/WW/EQ/NVT_100K/tert_cont_var.txt')
meq.read_mean_distances('/u3/WW/EQ/NVT_100K/distance.txt')
meq.calculate_bonded_rates()
np.save(working_dir + 'bonded_rates_matrix.npy', meq.bonded_rates_matrix)
meq.calculate_polar_rates(meq.tertiary_contact_residues)
np.save(working_dir + 'polar_rates_matrix.npy', meq.polar_rates_matrix)
meq.calculate_transition_rate_matrix()
#meq.calculate_E_init(top_path=root_path + 'ww_24_13_NEQ_', traj_path=root_path + 'ww_24_13_NEQ_prot_', iterator=range(200)) 
#np.save(os.path.expanduser('~/HiWi/WW/working_files/E_init_910K.npy'), meq.E_init)
#E_init_MD = np.load(os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_resides_149.npy'))[:,0]
E_init_MEQ = np.load(os.path.expanduser('~/HiWi/WW/working_files/E_init_910K.npy'))
meq.run_meq_simulation(E_init=E_init_MEQ, dt=4e-15, n_steps=25000)
