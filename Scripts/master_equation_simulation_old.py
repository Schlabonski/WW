#!/usr/bin/python3

class MasterEquationSimulation(object):

    """Small class to create and run master equation simulations."""

    def __init__(self,p_init, T_tau):
        """Initiation with initial probabilities and transition matrix.

        :p_init: vector of len N, initial propability distribution
        :T_tau: NxN matrix, transition propability rate matrix

        """
        self._p_init = p_init
        self._T_tau = T_tau

    def run(self, time_step, method, N_steps=None, exactitude=None):
      """Starts the simulation run.

      :time_step: float, time interval in which T_tau*p_init is evaluated
      :method: string, either 'convergence' or 'steps'
      :N_steps: int, if method is steps simulation will run for N_steps steps
      :exactitude: float, if method is convergence, simulation will run untill 
                   relaxed to exactitude
      :returns: time_evolution of every element of p_init

      """
      if method == 'steps':
        time_evolution = self.run_steps(time_step, N_steps)

      elif method == 'convergence':
        time_evolution = self.run_convergence(time_step, exactitude)

      return time_evolution
        
