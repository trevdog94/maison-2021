# -*- coding: utf-8 -*-
"""
==============================================================
Microscopic Markov Chain Approach for Measuring Disease Spread
==============================================================
"""

import numpy as np
import math

class mmcaCOVID:
    """

    The mmcaCOVID module implements a set of functions needed to model a Microscopic Markov Approach for measuring the spread of COVID19 infection as a function of 
    human mobility and social distancing parameters as well us the infectious seeds in a given patch. The model permits the study of various intervention
    scenarios like lockdowns and reopenings. The model is agnostic to the level of geographic resolution and can even be extended to study other diseases.

    """

    def __init__(self, R_ls = [np.random.rand(2,2)], epi_params = {'beta': 0.06, 'sigma': 1/5.1, 'gamma':1/21, 'omega':0.013, 'delta': 1/17.8}, I_0 = np.array([[1, 0]]), c_ls = [10 * np.random.rand(1, 2)], n_vec = np.array([[100, 100]])):
        """

        Parameters
        ----------
        R_ls : (list)

            A list of N x N mobility matrices that each represent a snapshot of the mobility distributions for each day in the fitting period and N represents the number of regions in the model.

        epi_params : (dict)

            A dictionary of the epidemiological parameters of the model.
        
        I_0 : (np.array)

            An array 1 x N array of the initial infectious seeds for each patch in the model.

        c_ls : (list)

            A list of 1 x N arrays where each element of the array represents the average contacts within a given patch.

        n_vec : (np.array)

            A 1 x N array where each element of the array represents the population size within a given patch.
        
        """

        ## Check for valid inputs
        assert len(R_ls) == c_ls, "The temporal extent of mobility matrices must match that of average contacts."
        
        ## Initialze number of patches
        self.N = len(initials)

        ## Initialize the fitting period
        self.timesteps = len(R_ls) 

        ## Initialize demographic parameters
        self.M_ls = M_ls 
        self.c_ls = c_ls
        self.n_ls = n_ls

        ## Initialize epidemiological parameters
        self.beta = epi_params['beta']
        self.sigma = epi_params['sigma']
        self.gamma = epi_params['gamma']
        self.omega = epi_params['omega']
        self.delta = epi_params['delta']

    def moving_population(R_kj, rho_I_k, n_k):
        """
        Calculate the number of infected individuals moving from one patch to the next for a snapshot in time.

        Parameters
        ----------
        R_kj : (float)

            Proportion of individuals in patch k who mobilize to patch j

        rho_I_k : (float)

            Proportion of infected individuals in patch k

        n_k : (int)

            Population in patch k
        
        Returns
        -------
        n_I_move : (int)

            Number of infected individuals from patch k who move to region j

        """
        n_I_move = n_k * rho_I * R_kj
        return floor(n_I_move)

    def update_Pi(self, beta, R, rho_I, c_vec, n_vec, patch_ids): 
        """

        Update the probability of infection in each patch according to the following formula:

        .. math::

            Pi_i(t) = \sum_{j=1}^N R_{ij}(t) { 1 - (1-\beta)^{c_j(t)/n_j^{eff}\sum_{k=1}^N(n^{I}_{k\to j}(t))}}

        Parameters
        ----------
        beta : (float)
            
            Infectivity rate
        
        R : (np.array)

            N x N mobility matrix such that R[i, j] denotes the proportion of individuals from region i who mobilize to
            region j at time t

        rho_I : (np.array)

            An array where each element of the array represents the proportion of infectious individuals within a given patch 

        c_vec : (np.array)

            An array where each element of the array represents the average contacts within a given patch

        n_vec : (np.array)

            An array where each element of the array represents the population size within a given patch
        
        patch_ids : (list)

            A list of patch id's for each patch in the multi-patch

        Returns
        -------
        df : (pd.DataFrame())

            A pandas dataframe with three columns: (patch_id, Pi, n_eff) 

        Example
        -------

        """
        ## Initialize number of patches
        N = self.N
        
        ## Initialize tracking vectors
        Pi = np.zeros(N)
        n_eff = np.zeros(N)

        ## Iterate through each resident patch
        for i in range(N):

            ## Iterate through each source patch where imported cases could have arisen
            for j in range(N):

                n_kj_sum = 0
                n_j_eff = 0

                ## Sum over all infectious populations traveling from patch k to patch j
                for k in range(N):
                    n_kj_sum += moving_population(R[k,j], rho_I[k], n_vec[k])
                    n_j_eff += n_vec[k] * R[k, j]
                # End for k

                ## Exponent for the probability of infection (Pi)
                exponent = (c_vec[i], * n_kj_sum / n_j_eff)
                Pi[i] += (R[i,j] * (1-pow((1-beta), exponent)))
                n_eff[j] = n_j_eff
            # End for j
        # End for i
        
        ## Update the probability of infection
        self.Pi = Pi

        ## Return a dataframe of Pi for each patch
        df = pd.DataFrame()
        
        if patch_ids is not None:
            df['patch_id'] = patch_ids
        # End if
        df['Pi'] = Pi
        df['n_eff'] = n_eff
        return df

    def compartment_evolution(self):
        """
        
        Calculate the next days compartment counts according to:

        .. math::

            \rho^S_(t+1) = (1-\Pi_i(t)) \rho^S_i(t)
            
            \rho^E_(t+1) = \Pi_i(t) \rho^S(t) + (1 - \sigma) \rho^E_i(t)
            
            \rho^I(t+1) = \sigma \rho_i^E(t) + (1 - \gamma (1-\omega)-\omega \delta) \rho^I_i(t))
            
            \rho^R(t+1) = (1 - \omega) \gamma \rho^I_i(t)

            \rho^D(t+1) = \omega \delta \rho^I_i(t)

        Parameters
        ----------


        Returns
        -------
       
        
        """
        ## Update the compartment estimates for each patch
        for i in range(N):
            self.rho_S[i] = (1 - self.Pi[i]) * self.rho_S[i]
            self.rho_E[i] = self.Pi[i] * self.rho_S[i] + (1-self.sigma) * self.rho_E[i]
            self.rho_I[i] = self.sigma * self.rho_E[i] + (1 - self.gamma * (1 - self.omega) - self.omega * self.delta)
            self.rho_R[i] = (1 - self.omega) * self.gamma * self.rho_I[i]    
            self.rho_D[i] = (1 - self.omega * self.delta * self.rho_I[i])

        return self.rho_S, self.rho_E, self.rho_I, self.rho_R, self.rho_D
        
    def run_epidemic_spreading(self, E_0, I_0, R_ls, beta, c_ls, n_vec, sigma, gamma, delta, omega):
        """
        Kicks off the MMCA model where each day the probability of infection is updated according to a Markov
        process which in turn governs the transition from susceptible individuals into the pre-infectious stage.

        Parameters
        ----------
        E_0 : (np.array)

            The initial pre-infectious seeds

        I_0 : (np.array)

            The initial infectious seeds

        R_ls : (list)

            A list of N x N mobility matrices that each represent a snapshot of the mobility distributions for each day in the fitting period and N represents the number of regions in the model.

        beta : 



        """
        timesteps = self.timesteps

        # for tau in range(1, timesteps):

    # def fit_model(fixed_params = []):
    #     return None

    # def update_output(self):
    #     return None