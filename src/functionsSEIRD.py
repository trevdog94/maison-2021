# This script contains a set of functions that walk through the model for spread of disease
# Some functions are for getting data for infection rates
# Would like this to also contain a Markov-model for future propagation
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta


###############################################################################
## Use: Run a Markov SEIRD Model With Dynamic Pi                             ##
## Inputs:                                                                   ##      
##      initialsdf = initial conditions data frame loc/S/E/I/R/D/date        ##
##      Rlist      = list of numpy array mobility matrices R[tau][i,j] is the## 
##                   proportion of individuals moving from i to j at time tau##     
##      beta       = infectivity rate                                        ##
##      cvlist     = list of vectors cvlist[tau][i] is the average number of ##
##                   contacts for someone in region i at time tau            ##
##      nvec       = list of populations (nvec[i] is pop in region i)        ##
##      sigma      = latent rate                                             ##
##      gamma      = recovery rate                                           ##
##      delta      = death rate                                              ##
##      omega      = fraction of cases that result in death                  ##
## Output:                                                                   ##
##      pandas data frame with loc/S/E/I/R/D/date/Pi values                  ##
###############################################################################
def executeSEIRDmobility(initialsdf, Rlist, beta, cvlist, nvec, sigma, gamma, delta, omega): 
  ## Step through the model with dynamic Pi
  timesteps1 = len(Rlist)
  timesteps2 = len(cvlist)
  N1 = len(Rlist[0])
  N2 = len(initialsdf[['loc']])
  if(N1 != N2):
    print('mobility matrices must have same rows/columns as number of communities reflected in initialsdf')
    return
  if(timesteps1 != timesteps2):
    print('number of matrices in Rlist and number of vectors in cvlist must be the same')
    return 
  timesteps=timesteps1
  N = N1
  startdate = initialsdf['date'][0]
  if isinstance(initialsdf['date'][0],str):
      initialsdf['date'] = datetime.datetime.strptime(initialsdf['date'][0],'%Y-%m-%d')
  initials = initialsdf[['S','E','I','R','D']].values
  initials = initials.astype('float64')
  communities = initialsdf[['loc']]
  projections = initialsdf
  rhoI = initialsdf['I']/nvec
  Pi = getPi(beta,Rlist[0],rhoI,cvlist[0],nvec)['Pi']
  projections['Pi'] = Pi
  for tau in range(1,timesteps):
    rhoI = np.zeros(N)
    date = startdate + timedelta(days=tau)
    for i in range(N):
      nextday = stepmodel(initials[i], Pi[i], sigma, gamma, delta, omega)
      ## New initial conditions
      initials[i] = nextday[0].values.tolist()
      rhoI[i] = nextday[0]["I"]/sum(nextday[0])
      ## Tracking probability of infection
      ## Tracking vector
    initialsdftmp = pd.DataFrame(initials)
    Pi = getPi(beta,Rlist[tau],rhoI,cvlist[tau],nvec)['Pi']
    for i in range(N):
      new_row = {'loc':communities['loc'][i],'S':initials[i,0],
                 'E':initials[i,1],'I':initials[i,2],
                 'R':initials[i,3],'D':initials[i,4],
                 'Pi':Pi[i],'date':date}
      projections = projections.append(new_row,ignore_index=True)
  return projections

###############################################################################
## Use: Calculate the next days compartment counts                           ##
## Inputs:                                                                   ##                                                   
##      initials <- initial conditions of the model (S/E/I/R/D)              ##
##      Pi <- probability of infection                                       ##
##      sigma <- latent rate                                                 ##
##      gamma <- recovery rate                                               ##
##      delta <- death rate                                                  ##
##      omega <- fraction of cases that result in death                      ##
## Output:                                                                   ##
##      stepmodel <- compartment counts for S/E/I/R/D                        ##
###############################################################################
def stepmodel(initials, Pi, sigma, gamma, delta, omega):
  tmA = gettransition(Pi, sigma, gamma, delta, omega)
  tmA = tmA.transpose()
  # create the DTMC
  cond = ["S","E","I","R","D"]
  initialsdf = pd.DataFrame(initials,index=cond)
  #dtmcA <- new("markovchain",transitionMatrix=tmA, 
  #             states=cond, 
  #             name="COVID MarkovChain");
  step = tmA.dot(initialsdf) #tmA @ initialsdf
  return step

###############################################################################
## Use: Calculate the transition matrix of the system                        ##
## Inputs:                                                                   ##
##      Pi <- probability of infection for a region                          ##
##      sigma <- latent rate                                                 ##
##      gamma <- recovery rate                                               ##
##      delta <- death rate                                                  ##
##      omega <- fraction of cases that result in death                      ##
## Output:                                                                   ##
##      gettransition <- a transition matrix for SEIRD model                ##
###############################################################################
def gettransition(Pi, sigma, gamma, delta, omega):
  # Define the transition matrix
  cond = ["S","E","I","R","D"]
  tmA = [[1-Pi, Pi, 0, 0, 0],
         [0, 1-sigma, sigma, 0, 0],
         [0, 0, 1-gamma*(1-omega)-omega*delta, gamma*(1-omega), omega*delta],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]]
  dfA = pd.DataFrame(tmA,columns=cond,index=cond)
  return dfA

###############################################################################
## Use: Find probability of infection in each region at a single time point  ##
## Inputs:                                                                   ##
##      beta  <- numeric, infectivity rate                                   ##
##      R     <- numpy array so that R[i,j] denotes the proportion           ##
##		           of individuals from region i who mobilize to region j at    ## 
##               time t                                                      ##
##      rhoI  <- vector for proportion of infected in each patch             ##
##      cvec  <- vector of average contacts for each patch                   ##
##      nvec  <- vector of population size for each patch                    ##
##      communities  <- vector of patch names; if None, this is ignored      ##
## Outputs: 	                                                               ##
##	Pi   <- vector such that Pi[i] provides the probability                  ##
##          someone in region i gets infected                                ##
##  neff <- vector of effective popultation for each region                  ##
###############################################################################
# Pi_i(t) = \sum_{j=1}^N R_{ij}(t) { 1 - (1-beta)^{c_j(t)/n_j^{eff}\sum_{k=1}^N(n^{I}_{k\to j}(t))}}
def getPi(beta,R,rhoI,cvec,nvec,communities = type(None)):
    N = len(R)
    Pi = np.zeros(N)
    neff = np.zeros(N)
    for i in range(N):
        for j in range(N):
            nkjsum = 0
            njeff = 0
            for k in range(N):
                nkjsum += movingpop1(R[k,j],rhoI[k],nvec[k])
                njeff += nvec[k]*R[k,j]
            #endfork
            exponent = (cvec[i] * nkjsum / njeff)
            Pi[i] += (R[i,j] * (1 - pow((1 - beta),exponent)))
            neff[j] = njeff
        #endforj
    #endfori
    df = pd.DataFrame()
    if communities is not None:
      df['loc'] = communities
    #endif
    df['Pi'] = Pi
    df['neff'] = neff
    return df

###############################################################################
## Use: Helper function: get number of infected individuals moving from      ##
##                       one region to another at a particular time-point    ##
## Inputs:                                                                   ##
##      Rkjt  <- the proportion of individuals from region k who mobilize to ##
##               region j at time t                                          ## 
##      rhoIt <- proportion of infected in patch (k at time t)               ##
##      n     <- population in patch (k)                                     ##
## Outputs: 	                                                               ##
##  nImove <- number of infected individuals from region k moving to region j##
##            at time t                                                      ##
###############################################################################
def movingpop1(Rkjt,rhoIt,n):
    nImove = n*rhoIt*Rkjt
    return nImove
