#!/usr/bin/python3

import os
import json
import h5py


import numpy as np
import subprocess as sub
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

#rc_params = {'lines.linewidth':1.1, 'text.latex.preamble': [r'\usepackage{siunitx}', r'\sisetup{detect-all}', r'\renewcommand*\sfdefault{lcmss}', r'\usepackage{sansmath}', r'\sansmath'], 'text.usetex':True}
rc_params = {'lines.linewidth':1.5, 'text.usetex':True}


palette = sns.color_palette(sns.xkcd_palette(['denim blue','orange red', 'golden', 'medium green','fuchsia', 'aquamarine', 'burnt sienna']), n_colors=10)
sns.set(style='ticks', font='serif', palette='Set1', context='paper', font_scale=1.4, rc=rc_params)


class Trajectory(object):

  """This class defines the object of a gromacs trajectory"""

  def __init__(self, traj_path, basename, basename_trr=None, tracking_dir=os.path.expanduser('~/HiWi/WW/tracking/')):
    """Init sets the path in which the trajectory is found and will define
    respective files for python

    :traj_path: path of trajectory

    """
    if basename_trr == None:
      basename_trr = basename

    if traj_path[-1] != '/':
        traj_path += '/'
    self._traj_path = traj_path

    self.basename = basename
    
# we set the paths to some oftne used files as attributes
    if os.path.isfile(traj_path+basename_trr+'.trr'): 
      self._trr = traj_path + basename_trr + '.trr'    
    else:
      print('{0} does not exist.'.format(traj_path+basename_trr+'.trr'))
    if os.path.isfile(traj_path+basename+'.gro'):
      self._gro = traj_path + basename + '.gro'
    if os.path.isfile(traj_path+basename+'.tpr'):
      self._tpr = traj_path + basename + '.tpr'

# the keep_track_dic keeps track of all files that were already generated for
# this trajectory and stores them in ahuman readable json file
    self.tracking_path = tracking_dir + basename + '.json'
    try:
      with open(self.tracking_path, 'r') as f:
        self._keep_track_dic = json.load(f)
    except:
      with open(self.tracking_path, 'w+') as f:
        json.dump({}, f)
    
  def load_keep_track(self):
    """This function reloads the keep_track dictionary from disk
    """
    with open(self.tracking_path, 'r') as f:
      self._keep_track_dic = json.load(f) 
  
  def dump_keep_track(self):
    """Dumps the current keep track dic to disk
    """
    with open(self.tracking_path, 'w+') as f:
      json.dump(self._keep_track_dic, f)

  def set_working_dir(self, working_dir):
    """This functions sets the working directory for the data that is produced
    from Trajectory
    """
    self.load_keep_track()
    if working_dir[-1] != '/':
      working_dir += '/'
    self._keep_track_dic['working_dir'] = working_dir
    self.dump_keep_track()

  def set_plot_dir(self, plot_dir):
    """This function sets the directory, where the created plots are stored
    """
    self.load_keep_track()
    self._keep_track_dic['plot_dir'] = plot_dir
    self.dump_keep_track()

  def analyse_velocities(self, outname=None):
    """Writes out the functions to a file in the working directory
    """
    if outname == None:
      outname = self.basename+'_velocities'
    self.load_keep_track()
    attributes = self._keep_track_dic
    working_dir = attributes['working_dir']
    velocities_output_path = working_dir + outname
    
    print('Writing velocities from {0} and {1}...'.format(self._trr, self._tpr))

    Process = sub.Popen(['g_traj', '-f', self._trr, '-s', self._tpr, '-ov', velocities_output_path], stdin=sub.PIPE)
    Process.communicate(bytes('1 \n', 'UTF-8'))

    self._keep_track_dic['velocities_xvg_path'] = velocities_output_path
    self.dump_keep_track()

  def atom_masses(self, top_path):
    """Gets the mass per atom from topology and saves them in numpy file

    :top_path: path to top file

    """
    self.load_keep_track()
    attributes = self._keep_track_dic
    working_dir = attributes['working_dir']
    
    print('Loading masses from {0}...'.format(top_path))

    with open(top_path) as f:
      topology = f.readlines()

    firstline = [s for s in topology if 'atoms' in s]
    firstline_index = topology.index(firstline[0])

    lastline = [s for s in topology if 'bonds' in s]
    lastline_index = topology.index(lastline[0])

    atom_list = topology[firstline_index:lastline_index]

    atom_masses = []
    for line in atom_list:
      words = line.split()
      try:
        atom_index = float(words[0])
        atom_mass = float(words[7])
        atom_masses.append([atom_index, atom_mass])
      except:
        pass

    atom_masses = np.array(atom_masses)

    masses_path = working_dir + self.basename + '_masses.npy'
    np.save(masses_path, atom_masses)
    self._keep_track_dic['atomic_masses'] = masses_path

    self.dump_keep_track()
    print('Stored atom masses in {0}.'.format(masses_path))

  def velocities_to_npy(self):
    """Reads .xvg output for velocities and writes the velocities to npy file.

    """
    self.load_keep_track()
    attributes = self._keep_track_dic
    working_dir = attributes['working_dir']
    velocities_xvg_path = attributes['velocities_xvg_path'] + '.xvg'
    
    print('Reading {0}...'.format(velocities_xvg_path))
    with open(velocities_xvg_path) as f:
      vel_xvg = f.readlines()
   
    print('Extracting velocities...')
    vel_xvg = [s for s in vel_xvg if s[0]!='#']
    legend = [s for s in vel_xvg if s[2]=='s']
    velocities = [s for s in vel_xvg if not (s[0]=='#' or s[0]=='@')]
    del vel_xvg
    n_timesteps = len(velocities)
    n_coordinates = len(velocities[0].split())
    print('{0} time steps, {1} atoms...'.format(n_timesteps, (n_coordinates-1)/3))
    n_algorithmsteps = len(velocities)
    velocities_arr = np.empty((n_timesteps, n_coordinates))
    print('Storing velocities in numpy array...')
    
# put all data into numpy array
    for i in range(len(velocities_arr)):
      velocities_arr[i,:] = np.array(velocities[i].split())
      print('{0:3.2f}% done...'.format((i/len(velocities_arr)*100)), end='\r')
    
    del velocities

    time_array = velocities_arr[:,0]
    print(time_array)
    velocities_arr = velocities_arr[:,1:]
    velocities_arr_rearranged = np.empty((int((n_coordinates-1)/3), 3, n_timesteps))
     
    print('Rearranging array...')
    for i in range(int((n_coordinates-1)/3)):
      velocities_arr_rearranged[i,0,:] = velocities_arr[:,i]
      velocities_arr_rearranged[i,1,:] = velocities_arr[:,i+1]
      velocities_arr_rearranged[i,2,:] = velocities_arr[:,i+2]

# construct an indexing array for the atoms
    print('Building atom index legend...')
    legend_arr = np.zeros(len(legend))
    for i in range(len(legend_arr)):
      legend_arr[i] = float(legend[i].split()[4])

# store created data in h5 file and update tracking dic
    np_savename = working_dir + self.basename + '_velocities.h5'
    print('Storing numpy array in {0}...'.format(np_savename))
    h5file = h5py.File(np_savename, 'w')
    h5file.create_dataset('velocities', data=velocities_arr_rearranged)
    h5file.create_dataset('time', data=time_array)
    h5file.create_dataset('atom_indices', data=legend_arr)
    h5file.close()

    self._keep_track_dic['velocity_arr_path'] = np_savename
    self.dump_keep_track()
    
  def kinetic_energies(self):
    """Calculates kinetic energies from masses and velocities by atom.


    """
# first load all important data from files
    self.load_keep_track()
    attributes = self._keep_track_dic

    working_dir = attributes['working_dir']
    vel_path = attributes['velocity_arr_path']
    mass_path = attributes['atomic_masses']
    
    print('Loading velocities and masses from {0}...'.format(vel_path))
    f = h5py.File(vel_path, 'r')
    velocities = f['velocities'][:]
    time = f['time'][:]
    a_ind = f['atom_indices'][:]
    f.close()
    a_ind = a_ind[0::3] # reduce indices to 1D
    masses = np.load(mass_path)
    
# iterate over velocity array to calculate sum squared velocities
    print('Calculating kinetic energies...')
    v_square = velocities**2
    v_square_sum = np.sum(v_square[:], axis=1)
    del v_square
    for i in range(len(v_square_sum)):
      v_square_sum[i] *= masses[(np.where(masses[:,0] == a_ind[i])[0]), 1]

# save data in h5 file
    savename = working_dir + self.basename + '_E_kin.h5'
    print('Saving kinetic energies to {0}...'.format(savename))
    h5file = h5py.File(savename, 'w')
    h5file.create_dataset('E_kin', data=v_square_sum)
    h5file.create_dataset('time', data=time)
    h5file.create_dataset('atom_indices', data=a_ind)
    h5file.close()

    self._keep_track_dic['E_kin_path'] = savename
    self.dump_keep_track()

class Statistical_ensemble(object):

  """Represents an ensemble of trajectories from which we can extract means"""

  def __init__(self, path, basename, iterator, basename_trr=None, basename_ensemble='ensemble', tracking_dir=os.path.expanduser('~/HiWi/WW/tracking/'), working_dir=os.path.expanduser('~/HiWi/WW/working_files/')):
    self.basename = basename
    self.iterator = iterator
    self.working_dir = working_dir
    self.basename_ensemble = basename_ensemble

    ensemble = []
    for i in iterator:
      traj = Trajectory(path, basename+str(i), basename_trr=basename_trr+str(i))
      ensemble.append(traj)

    self.ensemble = ensemble

# the keep_track_dic keeps track of all files that were already generated for
# this trajectory and stores them in ahuman readable json file
    self.tracking_path = tracking_dir + basename_ensemble + '.json'
    try:
      with open(self.tracking_path, 'r') as f:
        self._keep_track_dic = json.load(f)
    except:
      with open(self.tracking_path, 'w+') as f:
        json.dump({}, f)

  def load_keep_track(self):
    """This function reloads the keep_track dictionary from disk
    """
    with open(self.tracking_path, 'r') as f:
      self._keep_track_dic = json.load(f) 
  
  def dump_keep_track(self):
    """Dumps the current keep track dic to disk
    """
    with open(self.tracking_path, 'w+') as f:
      json.dump(self._keep_track_dic, f)
  
  def E_kin_evolution(self, residues):
    self.load_keep_track()
    #time_rebin = np.linspace(time_interval[0], time_interval[1], time_steps)
    E_kin_res_i = True
    time_res_i = None
    residue_names = []
    
    for i_res, res in enumerate(residues):
      print('Calculating ensemble average for residue {0}...'.format(res[0]))
      for traj in self.ensemble:
        print('Analysing trajectory {0}...'.format(traj.basename))
        traj.load_keep_track()
        E_kin_path = traj._keep_track_dic['E_kin_path']

# load data from h5 files
        f = h5py.File(E_kin_path, 'r')
        E_kin = f['E_kin'][:]
        time = f['time'][:]
        a_ind = f['atom_indices'][:]
        f.close()
# use average kinetic energy over residue
        E_kin_indices = [np.where(at_index==a_ind)[0][0] for at_index in a_ind if at_index in res[1:]]
        E_kin = E_kin[E_kin_indices]
        E_kin = E_kin.sum(axis=0)
        
        #plt.plot(np.arange(len(E_kin)), E_kin)
        #plt.show()
# rebin the data according to the given time scale
# nope we don't rebin now!
        '''
        E_kin_rebin = np.zeros(time_steps)
        for i in range(time_steps-1):
          E_kin_rebin[i] = np.mean(E_kin[np.where((time>time_rebin[i]) and (time < time_rebin[i+1]))[0]])
        '''
         
# store data in array of residues
        if type(E_kin_res_i)== bool:
          E_kin_res_i = np.zeros((len(residues), len(E_kin)))
          time_res_i = np.zeros((len(residues), len(time)))
          E_kin_res_i[i_res] += E_kin
          time_res_i[i_res] += time

        elif type(E_kin_res_i) == np.ndarray:
          E_kin_res_i[i_res] += E_kin
          time_res_i[i_res] += time

# out of ensemble loop, we average over whole ensemble and append residues name
# to list
      E_kin_res_i[i_res] /= len(self.ensemble)
      time_res_i[i_res] /= len(self.ensemble)
      residue_names.append(res[0])


# out of residue loop, kinetic energies per residue are stored in .h5 file
    
    savename = self.working_dir + self.basename_ensemble + '_E_kin_per_residue.h5'
    print('Saving kinetic energies to {0}...'.format(savename))
    h5file = h5py.File(savename, 'w')
    for i in range(len(residue_names)):
      h5file.create_dataset(residue_names[i], data=E_kin_res_i[i])
      h5file.create_dataset(residue_names[i]+'_time', data = time_res_i[i])
      print(time_res_i[i])
    h5file.close()

    self._keep_track_dic['E_kin_per_residue_path'] = savename
    self.dump_keep_track()

  def plot_E_kin(self, residues, plot_dir=os.path.expanduser('~/HiWi/WW/Plots/')):
    """Plots the kinetic energy of residues in one plot

    :residues: list of names of residues
    :returns: TODO

    """
# load file locations
    self.load_keep_track()
    E_kin_savepath = self._keep_track_dic['E_kin_per_residue_path']

# create matplotlib figure
    fig = plt.figure(figsize=(8,6))
    axe = fig.add_subplot(111)
   
# iterate over all residues
    for res_name in residues:
      
# load E_kin and time from h5
      h5file = h5py.File(E_kin_savepath, 'r')
      E_kin = h5file[res_name][:]
      time = h5file[res_name+'_time'][:]
      h5file.close()

# apply filter 
      E_kin = savgol_filter(E_kin, 31,3 )

      axe.plot(time, E_kin, label=res_name)
    
# set plot parameters
    axe.set_xscale('log')
    axe.set_xlim(xmin= 0.1, xmax=500)
    axe.set_xlabel('t in ps')
    axe.set_ylabel(r'E / $\frac{kJ}{mol}$')
    axe.legend(loc=0)
    plt.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left='on',      # ticks along the bottom edge are off
      right='off',         # ticks along the top edge are off
      labelright='off') # labels along the bottom edge are off
    plt.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='on',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      labelbottom='on') # labels along the bottom edge are off
    axe.grid()
    
    plt.show()

# save the figure
    savename = plot_dir + 'E_kin_residues_{0}'.format(residues[:])
    fig.savefig('{0}.pdf'.format(savename), dpi=100)
    fig.savefig('{0}.png'.format(savename), dpi=100)

    
