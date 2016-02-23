import os

try:
    import MDAnalysis as md
except Exception as e:
    print(e)
    print('Functions using MDAnalysis will not be available.')
import numpy as np
import itertools as it

class MEQ_Simulation(object):
  """class to run MEQ simulation for energy transport in proteins"""
  
  D_B = 1.25 * 1e-18 * 1e12 # diffusion constant for bonds in nm^2 ps^-1
  D_C_k_B_T = 1.1e-4 * 1e-18 * 1e12 # diffusion constant for polar interactions
  
  k_to_solvent = 1/5.6e-12
  k_from_solvent = 1/240e-12
  
  k_B = 1.38064852e-23 # J/K

  def __init__(self, top_path, traj_path):
    """inits object basically with properties of MD.universe .

    :top_path: path to topology file
    :traj_path: path to trajectory

    """
    self._top_path = top_path
    self._traj_path = traj_path
    
    un =  md.Universe(self._top_path)
    self.residues = [res for res in un.residues if res.name not in ['SOL', 'CL']]

    self.mean_distance_matrix = None
    self.distance_variance_matrix = None

  def calculate_E_init(self, top_path, traj_path, iterator):
    """Calculate the initial energies of residues and water.

    :top_path: str, root path of the top files
    :traj_path: str, root path of trajectories
    :iterator: iterator to be appended to path names

    """
    n_energies = len(self.residues) + 1 # +1 for solvent energy
    E_init = np.zeros(n_energies)
    
    print('Calculating initial energies...')

    # first we will iterate over all trajectories to  obtain the mean initial 
    # kinetic energies of the residues
    for i in iterator:
      top = top_path + str(i) + '.gro'
      trr = traj_path + str(i) + '.trr'
      un = md.Universe(top, trr)
      
      # use residues of interest (no solvent)
      residues = [res for res in un.residues if res.name not in ['SOL', 'CL']]
      
      # calculate kinetic energy of respective residues sum 1/2 mv^2
      for timestep in un.trajectory:
        if timestep == 1:
          for n, res in enumerate(residues):
            vel_squared = np.sum((res.get_velocities())**2, axis=1)
            masses = res.masses
            E_kin = np.sum(0.5 * vel_squared * masses)
            E_init[n] += E_kin / len(iterator)

        elif timestep > 1:
          break

      print('Analyzed %i out of %i trajectories...' %(i+1, max(iterator)+1))

    # set the solvent energy, assuming that for each water E_kin = 3 k_B T 
    # with T=10K, because solvent molecules have 6 dof (model). Multiply by
    # number of molecules N = 1410

    E_init[-1] = 1410 * 3* 10 * self.k_B /1000 * 6.02214129e23 # last factor for kJ/mole

    self.E_init = E_init

  def calculate_degrees_of_freedom(self, residue):
    """Puts out the degrees of freedom for a given residue if H-bonds are restricted
    :residue: type MDAnalysis.residue
    :returns: degrees of freedom

    """
    N_H = 0
    N_atoms = 0
    for atom in residue.atoms:
      N_atoms += 1
      if atom.type == 'H':
        N_H += 1
    return 3*N_atoms - N_H
      
  def calculate_bonded_rates(self):
    """Calculates the energy transmission rates between bonded backbone residues

    """
    residues = self.residues
    
    # create nxn Matrix (n = len(residues)) to store rates of bonded interaction
    k_ij_bonded = np.zeros((len(residues),len(residues)))
    indices = np.arange(len(residues))

    # now iterate over all neighboured pairs of indices to create off diagonal k_ij
    # assume that distance matrix exists as self.mean_distance_matrix
    mean_dist_mat = self.mean_distance_matrix
    for n in indices[:-1]:
      i,j = indices[n], indices[n+1]
      f_i = self.calculate_degrees_of_freedom(residues[i])
      f_j = self.calculate_degrees_of_freedom(residues[j])
      k_ij_bonded[i,j] = self.D_B/mean_dist_mat[i,j]**2*np.sqrt(f_j/f_i)
      k_ij_bonded[j,i] = self.D_B/mean_dist_mat[j,i]**2*np.sqrt(f_i/f_j)

    self.bonded_rates_matrix = k_ij_bonded

  def calculate_polar_rates(self, res_list):
    """Calculates energy transmission rates for polar interactions (shortcuts)

    : res_list : list of lists, contains names of residues for which polar
                 interactions exist; each sublist should contain a pair of 
                 residues (inverse not necessary!)
    """
    residues = self.residues

    # create nxn matrix to store polar interaction rates
    k_ij_polar = np.zeros((len(residues), len(residues)))

    # iterate over pairs of residues to fill k_ij_polar
    for res_pair in res_list:
      res_i = res_pair[0]
      res_j = res_pair[1]
      i = residues.index(res_i)
      j = residues.index(res_j)
      f_i = self.calculate_degrees_of_freedom(residues[i])
      f_j = self.calculate_degrees_of_freedom(residues[j])

      # assume that self.distance_variance_matrix contains pairwise distance
      # variances of residues
      k_ij_polar[i,j] = self.D_C_k_B_T/self.distance_variance_matrix[i,j]*np.sqrt(f_j/f_i)
      k_ij_polar[j,i] = self.D_C_k_B_T/self.distance_variance_matrix[j,i]*np.sqrt(f_i/f_j)

    self.polar_rates_matrix = k_ij_polar

  def calculate_transition_rate_matrix(self):
    """This will calculate the transition rate matrix from the bonded rates and
    the polar rates. The last index of the transition rate matrix will 
    correspond to the solvent.

    """
    n_res = len(self.residues) # number of residues
    matrix_size = n_res + 1 # one more for solvent
    transition_rate_matrix = np.zeros((matrix_size, matrix_size))
    
    polar_rates_matrix = self.polar_rates_matrix
    bonded_rates_matrix = self.bonded_rates_matrix

    for i in range(matrix_size): # loss terms
      if i <= n_res - 1: # residue energies loss terms (mind indexing!!)
        transition_rate_matrix[i,i] -= np.sum(polar_rates_matrix[i])
        transition_rate_matrix[i,i] -= np.sum(bonded_rates_matrix[i])
        transition_rate_matrix[i,i] -= self.k_to_solvent

      elif i == n_res: # solvent energy loss terms
        transition_rate_matrix[i,i] -= n_res * self.k_from_solvent

    for (i,j) in it.combinations(range(matrix_size), 2): # gain terms
      if i <= n_res -1 and j <= n_res - 1: # residue to residue flow
        transition_rate_matrix[i,j] += bonded_rates_matrix[j,i]
        transition_rate_matrix[i,j] += polar_rates_matrix[j,i]
        transition_rate_matrix[j,i] += bonded_rates_matrix[i,j]
        transition_rate_matrix[j,i] += polar_rates_matrix[i,j]

      elif j == n_res: # protein to solvent case
        transition_rate_matrix[i,j] += self.k_from_solvent
      
      '''
      elif j == n_res: # protein from solvent case
        transition_rate_matrix[i,j] += self.k_from_solvent
      '''
    for i in range(n_res):
      transition_rate_matrix[-1,i] += self.k_to_solvent

    self.transition_rate_matrix = transition_rate_matrix

  def run_meq_simulation(self, E_init, n_steps=1000, dt=1e-9):
    """Run the master equation simulation based on initial energy distribution
    E_init and the transition rate matrix for n_steps steps with timestep dt.

    :E_init: array or vector, containing the initial energies of all residues
             and the solvent
    :n_steps: int, number of simulation steps
    :dt: float, simulation time step

    """
    trm = np.matrix(self.transition_rate_matrix * dt) # scaled with timestep
    matrix_size = len(self.residues) + 1
    energy_evolution = np.zeros((n_steps, matrix_size)) # to store energies
    energy_evolution[0,:] = np.matrix(E_init)
    energy_evolution = np.matrix(energy_evolution)
    
    for i in range(1, n_steps):
      energy_evolution[i,:] = energy_evolution[i-1,:] + (trm*energy_evolution[i-1,:].T).T
      if i%100 == 0:
        print(str(i/n_steps*100) + ' percent done...')

    energy_evolution = energy_evolution.T
    self.E_evolv = energy_evolution

    np.save(os.path.expanduser('~/HiWi/WW/working_files/910K/E_kin_all_residues_MEQ.npy'), energy_evolution)

  def read_mean_distances(self, path_to_table):
    """Read the mean distances between residues from table and put them into
       self.mean_distance_matrix

    :path_to_table: str, path to the txt file

    """
    # create mean_distance_matrix to store distances
    mean_dist_mat = np.zeros((len(self.residues), len(self.residues)))

    data = np.loadtxt(path_to_table)
    for line in data:
      i = int(line[0]-1) # indices are residue numbers, starting from one!!
      j = int(line[1]-1)
      dist = line[2]*1e-9 # values given in nanometer
      mean_dist_mat[i,j] = dist
      mean_dist_mat[j,i] = dist

    self.mean_distance_matrix = mean_dist_mat

  def read_distance_variances(self, path_to_table):
    """Same as mean distances.
    """
    # create distance variance matrix
    dist_var_mat = np.zeros((len(self.residues), len(self.residues)))

    data = np.loadtxt(path_to_table)
    tertiary_contact_residues = []
    for line in data:
      i = int(line[0]) # here we are given indices starting at 0!!
      j = int(line[1])
      dist_var = line[3]*1e-18 # given values in nm^2
      dist_var_mat[i,j] = dist_var
      dist_var_mat[j,i] = dist_var
      
      if (self.residues[i], self.residues[j]) not in tertiary_contact_residues:
        tertiary_contact_residues.append((self.residues[i], self.residues[j]))

    self.tertiary_contact_residues = tertiary_contact_residues
    self.distance_variance_matrix = dist_var_mat
