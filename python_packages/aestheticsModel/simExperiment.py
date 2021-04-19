# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9

@author: abrielmann

consolidated collection of functions needed to generate predictions of the
model of aesthetic value introduced by Brielmann & Dayan (2021)
"""
import numpy as np
from statistics import NormalDist # for 1D logLik

def logLik_from_mahalanobis(stim, mu_x, cov, k=None):
    """
    Calculate the log likelihood of the current image given the presumed
    'system state' mu_x and covariance matrix cov based on the mahalanobis
    distance between image feature vector and system state.
    This function does accept singular covariances matrices as input and will
    compute a pseudo-inverse in these cases.
    This function also calculates the log-likelihood for the 1D case as the 
    probability of the stimulus feature value based on the system state's PDF.
    """
    if k is None:
        k = 0

    stim = np.array(stim)
    mu_x = np.array(mu_x)
    
    if mu_x.shape==(1,) or mu_x.shape==(): # if 1D
        if cov>0:
            z = NormalDist(mu=mu_x, sigma=cov).pdf(stim)
        else:
            z = 0
            
        if z==0:
            log_p = np.log(1e-10)
        else:
            log_p = np.log(z)
    else:
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                inv_cov = np.linalg.pinv(cov)
            else:
                raise
        s_minus_mu = stim - mu_x
        log_p = k - np.dot(np.dot(s_minus_mu.T, inv_cov), s_minus_mu) / 2

    return log_p

def KL_distributions(mu_true, cov_true, mu_state, cov_state):
    """
    Compute the Kullback-Leibler divergence between two Gaussian distirbutions
     specified by the input parameters.
    See http://stanford.edu/~jduchi/projects/general_notes.pdf (last page) 
    for a derivation for the n-dimensional case.
    This function does accept singular covariances matrices as input and will
    compute a pseudo-inverse and determinant in these cases.
    this function also calculates KL divergence in a simplified manner 
    in case the input is 1-dimensional.

    Parameters
    ----------
    mu_true : array_like
        means of the 'true' feature dsitribution.
    cov_true : ndarray
        covariance matrix of the 'true' feature distribution
    mu_state : array_like
        means of the system state.
    cov_state : ndarray
        covariance matrix of the system state

    Returns
    -------
    KL.

    """
    mu_diff = np.array(mu_state).astype(float) - np.array(mu_true).astype(float)
    
    # 1D case
    if np.array(mu_true).shape==() or np.array(mu_true).shape==(1,):
        if cov_state==0:
            KL = np.inf
        else:
            KL = np.log(cov_state/cov_true) + (cov_true-mu_diff**2)/(2*cov_state**2) - 0.5
    # 2D or higher dimensional case
    else:
        n = len(mu_true)
        try:
            cov_state_inv = np.linalg.inv(cov_state)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                cov_state_inv = np.linalg.pinv(cov_state)
                eig_values,_ = np.linalg.eig(cov_state)
                cov_state_det = np.product(eig_values[eig_values > 1e-12])
            else:
                raise
        else:
            cov_state_det = np.linalg.det(cov_state)
    
        try:
            np.linalg.inv(cov_true)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                eig_values,_ = np.linalg.eig(cov_true)
                cov_true_det = np.product(eig_values[eig_values > 1e-12])
            else:
                raise
        else:
            cov_true_det = np.linalg.det(cov_true)
    
        KL = (0.5 *
              np.log((cov_state_det / cov_true_det))
              - n
              + np.trace(np.dot(cov_state_inv, cov_true))
              + np.dot(np.dot(mu_diff.T, cov_state_inv), mu_diff))

    return KL

def simulate_practice_trials(mu, cov, alpha, stim_mu, n_stims, stim_dur):
    """
    update agent's system state after exposure to n_stims stimuli from the
    stimulus distribution with means stim_mu for stim_dur each
    given an agent's system state as defined by mu and cov
    Returns updated mu.
    This function abbreviates the simulation process by updating mu only once
    summing up estimated changes.

    Parameters
    ----------
    mu : array_like
        list or 1d-array of length n_features; means of agent's system state
        before practice trials.
    cov : ndarrary
        array of size len(mu) x len(mu)
       covariance of agent's system state.
    alpha : float
         magnitude of shift in mu towards currently 'presented' feature vector.
    stim_mu : ndarray
        means of the stimulus distribution.
    n_stims : int
        number of stimuli the agent is exposed to.
    stim_dur: int
        number of time steps per stimulus exposure.

    Returns
    -------
    mu_new : array, float
        1d-array of length n_features; means of agent's system state
        after practice trials.

    """

    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # update mu
    s_minus_mu = stim_mu - mu_new
    sum_change = s_minus_mu * (1-(1-alpha) ** (stim_dur * n_stims))
    mu_new = mu_new + sum_change


    return mu_new


def calc_predictions(mu, cov, mu_true, cov_true, alpha,
                                   stim,
                                   w_r=1, w_V=1, bias=0,
                                   return_mu=False, return_r_t=False, 
                                   return_dV=False):
    """
    Calculates aesthetic value A for a given system state and stimulus 
    given known p_true as specified by mu_true and cov_true.

    Parameters
    ----------
    mu : array_like
        list or 1d-array of length n_features; means of agent's system state
        before practice trials.
    cov : ndarrary
        array of size len(mu) x len(mu)
       covariance of agent's system state
    mu_true : array_like
        list or 1d-array of length n_features; means of p_true
        before practice trials.
    cov_true : ndarrary
        array of size len(mu) x len(mu)
        covariance of p_true
    alpha : float
         magnitude of shift in mu towards currently 'presented' feature vector.
    stim : ndarray
        feature value combinations for the stimulus
    w_r : float, optional
        realtive weight of r(t) for calculating A(t). The default is 1.
    w_V : float, optional
        relative weight of delta-V for calculating A(t). The default is 1.
    bias : float, optional
        constant to be added to predicted A(t). The default is 0.
    return_mu : bool, optional
        whether to return updated value of mu. The default is False.
    return_r_t : bool, optional
        whether to return predicted r(t). The default is False.
    return_dV : bool, optional
        whether to return predicted Delta-V(t). The default is False.

    Returns
    -------
    A_t : float
        predicted A(t) for stim.
    mu_new : array, float, optional
        list or 1d-array of length n_features;
        means of agent's system state after exposure.
    r_t : float, optional
        predicted r(t) for stim.
    delta_V : float, optional
        predicted Delta-V(t) for stim.
    """
    if np.array(mu).shape==() or np.array(mu).shape==(1,):
        n_features = 1
    else:
        n_features = len(mu)
        
    mu_new = mu.copy()
    mu_new = np.array(mu_new).astype(float)

    # update mu because at this moment, agent is exposed to stim
    mu_new = mu_new + ((stim - mu_new) * alpha)

    # get r
    r_t = np.exp(logLik_from_mahalanobis(stim, mu_new, cov))

    # get V(t)
    V_t = n_features - KL_distributions(mu_true, cov_true, mu_new, cov)

    # expected mu
    mu_exp = mu_new + ((stim - mu_new) * alpha)

    # get estimate for V(t+1)
    V_t_exp = n_features - KL_distributions(mu_true, cov_true, mu_exp, cov)

    # get delta-V
    delta_V = V_t_exp - V_t

    A_t = bias + w_r * r_t + w_V * delta_V

    if return_mu:
        if return_r_t:
            if return_dV:
                return A_t, mu_new, r_t, delta_V
            else:
                return A_t, mu_new, r_t
        else:
            if return_dV:
                return A_t, mu_new, delta_V
            else:
                return A_t, mu_new
    elif return_r_t:
        if return_dV:
            return A_t, r_t, delta_V
        else:
            return A_t, r_t
    else:
        if return_dV:
            return A_t, delta_V
        else:
            return A_t
