# -*- coding: utf-8 -*-
"""
==============================================================
Microscopic Markov Chain Approach for Measuring Disease Spread
==============================================================
"""

import numpy as np
import pandas as pd
import datetime

class mmcaCOVID:
    """

    The mmcaCOVID module implements a set of functions needed to model a Microscopic Markov Approach for measuring the spread of COVID19 infection as a function of 
    human mobility and social distancing parameters as well us the infectious seeds in a given patch. The model permits the study of various intervention
    scenarios like lockdowns and reopenings. The model is agnostic to the level of geographic resolution and can even be extended to study other diseases.

    """

    def __init__(self, R_ls = [np.random.rand(2,2)], epi_params = {'beta': 0.06, 'sigma': 1/5.1, 'gamma':1/21, 'omega':0.013, 'delta': 1/17.8}, c_ls = [10 * np.random.rand(1, 2)], n_vec = np.array([[100, 100]]), patch_ids = list(range(2))):
        """

        Parameters
        ----------
        R_ls : (list)

            A list of N x N mobility matrices that each represent a snapshot of the mobility distributions for each day in the fitting period and N represents the number of regions in the model.

        epi_params : (dict)

            A dictionary of the epidemiological parameters of the model.

        c_ls : (list)

            A list of 1 x N arrays where each element of the array represents the average contacts within a given patch.

        n_vec : (np.array)

            A 1 x N array where each element of the array represents the population size within a given patch.
        
        patch_ids : (list)

            A list of patch id's for each patch in the multi-patch
        """
        ## Initialze number of patches
        self.N = len(n_vec)

        ## Initialize the fitting period
        self.tau = len(R_ls) 

        ## Check for valid inputs
        assert len(R_ls) == self.tau, "The temporal extent of mobility matrices must match that of average contacts."
        assert len(R_ls[0]) == self.N, "The dimensions of the mobility matrices do not match that of the population size array."
        assert len(c_ls[0]) == self.N, "The dimensions of the average contact arrays do not match that of the population size array."
        
        ## Initialize demographic parameters
        self.R_ls = R_ls 
        self.c_ls = c_ls
        self.n_vec = n_vec
        self.patch_ids = patch_ids

        ## Initialize epidemiological parameters
        self.beta = epi_params['beta']
        self.sigma = epi_params['sigma']
        self.gamma = epi_params['gamma']
        self.omega = epi_params['omega']
        self.delta = epi_params['delta']

    def moving_population(self, R_kj, rho_I_k, n_k):
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
        n_I_move = n_k * rho_I_k * R_kj
        return n_I_move

    def update_Pi(self, beta, R, rho_I, c_vec, n_vec): 
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
                    n_kj_sum += self.moving_population(R[k,j], rho_I[k], n_vec[k])
                    n_j_eff += n_vec[k] * R[k, j]
                # End for k

                ## Exponent for the probability of infection (Pi)
                exponent = (c_vec[i] * n_kj_sum / n_j_eff)
                Pi[i] += (R[i,j] * (1-pow((1-beta), exponent)))
                n_eff[j] = n_j_eff
            # End for j
        # End for i
        
        ## Update the probability of infection
        self.Pi = Pi

        ## Update the effective population
        self.n_eff = n_eff

        return Pi, n_eff

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
        ## Initialize number of patches 
        N = self.N

        ## Initialize numpy arrays with zeros
        rho_S = np.zeros(N)
        rho_E = np.zeros(N)
        rho_I = np.zeros(N)
        
        ## Update the compartment estimates for each patch
        for i in range(N):
            rho_S[i] = (1 - self.Pi[i]) * self.rho_S[i]
            rho_E[i] = self.Pi[i] * self.rho_S[i] + (1-self.sigma) * self.rho_E[i]
            rho_I[i] = self.sigma * self.rho_E[i] + (1 - self.gamma * (1 - self.omega) - self.omega * self.delta) * self.rho_I[i]
            self.rho_R[i] += (1 - self.omega) * self.gamma * self.rho_I[i]    
            self.rho_D[i] += self.omega * self.delta * self.rho_I[i]
        
        self.rho_S = rho_S
        self.rho_E = rho_E
        self.rho_I = rho_I

        return self.rho_S, self.rho_E, self.rho_I, self.rho_R, self.rho_D
    
    def set_seed(self, E_0, I_0, start_date):
        """
        Sets the initial fractions that exist in each compartment so that the Markov process can be kicked off

        Parameters
        ----------
        E_0 : (np.array)

            An array 1 x N array of the initial pre-infectious seeds for each patch in the model.

        I_0 : (np.array)

            An array 1 x N array of the initial infectious seeds for each patch in the model.
        
        Returns
        -------


        """
        ## Initialize model parameters
        tau = self.tau
        N = self.N
        c_ls = self.c_ls
        n_vec = self.n_vec
        beta = self.beta
        R_ls = self.R_ls
        n_vec = self.n_vec

        ## Grab the initial mobility matrix and contact vectors
        R = R_ls[0]
        c_vec = c_ls[0]

        ## Initialize numpy arrays with zeros
        rho_S = np.zeros(N)
        rho_E = np.zeros(N)
        rho_I = np.zeros(N)
        rho_R = np.zeros(N)
        rho_D = np.zeros(N)
        
        ## Update S, E, and I compartments to agree with infectious seeds
        for i in range(N):
            rho_E[i] = E_0[i] / n_vec[i]
            rho_I[i] = I_0[i] / n_vec[i]
            rho_S[i] = 1 - (rho_E[i] + rho_I[i])

        ## Update the initial conditions
        self.rho_S = rho_S
        self.rho_E = rho_E
        self.rho_I = rho_I
        self.rho_R = rho_R
        self.rho_D = rho_D

        ## Estimate the initial probability of infection
        self.update_Pi(beta, R, rho_I, c_vec, n_vec)

        ## Initialize the output dataframe
        out_df = pd.DataFrame()
        start_date = pd.to_datetime(start_date)
        self.start_date = start_date
        date = np.repeat(start_date, N, axis=0)

        out_df.loc[:, 'date'] = date
        out_df.loc[:, 'patch_id'] = self.patch_ids
        out_df.loc[:, 'rho_S'] = self.rho_S
        out_df.loc[:, 'rho_E'] = self.rho_E
        out_df.loc[:, 'rho_I'] = self.rho_I
        out_df.loc[:, 'rho_R'] = self.rho_R
        out_df.loc[:, 'rho_D'] = self.rho_D
        out_df.loc[:, 'Pi'] = self.Pi
        out_df.loc[:, 'n_eff'] = self.n_eff

        self.out_df = out_df

        return None


    def run_epidemic_spreading(self):
        """

        Kicks off the MMCA model where each day the probability of infection is updated according to a Markov
        process which in turn governs the transition from susceptible individuals into the pre-infectious stage.

        Parameters
        ----------

        Returns
        -------
        
        """
        ## Ensure infectious seeds have been set
        assert hasattr(self, 'rho_I'), "You must set the infectious seeds before running the model. Maybe try: mmca.set_seed(E_0, I_0)"

        ## Initialize model parameters
        tau = self.tau
        N = self.N
        c_ls = self.c_ls
        n_vec = self.n_vec
        beta = self.beta
        R_ls = self.R_ls
        n_vec = self.n_vec

        ## Step through the model
        for t in range(tau):
            # Retrieve current infectious fraction of the population
            rho_I = self.rho_I
            
            # Retrieve current mobility matrix
            R = R_ls[t]

            # Retrieve current average contacts vector
            c_vec = c_ls[t]

            # Update the next days compartment counts
            self.compartment_evolution()

            # Update the probability of infection
            self.update_Pi(beta, R, rho_I, c_vec, n_vec)

            # Update the output dataframe
            self.update_output(t)

            print("Timestep t: %s"%(t))


    def update_output(self, t):
        """
        Updates the output dataframe with all parameters from the current model step.

        Parameters
        ----------
        
        """
        N = self.N

        today = self.start_date + datetime.timedelta(t)
        today_df = pd.DataFrame()

        date = np.repeat(today, N, axis=0)

        today_df.loc[:, 'date'] = date
        today_df.loc[:, 'patch_id'] = self.patch_ids
        today_df.loc[:, 'rho_S'] = self.rho_S
        today_df.loc[:, 'rho_E'] = self.rho_E
        today_df.loc[:, 'rho_I'] = self.rho_I
        today_df.loc[:, 'rho_R'] = self.rho_R
        today_df.loc[:, 'rho_D'] = self.rho_D
        today_df.loc[:, 'Pi'] = self.Pi
        today_df.loc[:, 'n_eff'] = self.n_eff

        self.out_df = pd.concat([self.out_df, today_df])

        return None

    # def fit_model(fixed_params = []):
    #     return None