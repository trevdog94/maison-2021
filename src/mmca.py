# -*- coding: utf-8 -*-
"""
=======================
MMCA for Disease Spread
=======================
"""

import numpy as np

class mmcaCOVID:
    """

    The mmcaCOVID module implements a set of functions needed to model a Microscopic Markov Approach for measuring the spread of COVID19 infection as a function of 
    human mobility and social distancing parameters as well us the infectious seeds in a given patch. The model permits the study of various intervention
    scenarios like lockdowns and reopenings. The model is agnostic to the level of geographic resolution and can even be extended to study other diseases.

    Parameters
    ----------
    R_ls : (list)

        A list of N x N mobility matrices that each represent a snapshot of the mobility distributions for each day in the fitting period and N represents the number of regions in the model.

    epi_params : (dict)

        A dictionary of the epidemiological parameters of the model.
    
    initials_array : (np.array)

        An array 1 x N array of the initial infectious seeds for each patch in the model.

    c_ls : (list)

        A list of 1 x N arrays where each element of the array represents the average contacts within a given patch.

    n_array : (np.array)

        An 1 x N array where each element of the array represents the population size within a given patch.
    """

    def __init__(self, R_ls = [np.random.rand(2,2)], epi_params = {'beta': 0.06, 'sigma': 1/5.1, 'gamma':1/21, 'omega':0.013, 'delta': 1/17.8}, initals_array = np.array([[1, 0]]), c_ls = [10 * np.random.rand(1, 2)], n_array = np.array([[100, 100]])):
        
        ## Check for valid inputs
        assert len(R_ls) == c_ls, "The temporal extent of mobility matrices must match that of average contacts."
        
        ## Initialze number of patches
        self.N = len(initials_array)

        ## Initialize the fitting period
        self.t1 = len(R_ls) 

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

    def update_Pi(self): 
        """
        Update the probability of infection in each patch.

        Parameters
        ----------


        """

    