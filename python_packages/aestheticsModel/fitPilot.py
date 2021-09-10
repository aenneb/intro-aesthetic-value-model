# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:16:04 2021
Last updated on Wed Aug 11 11:38:00 2021

@author: abrielmann

Helper functions for fitting the data of the pilot experiment run in July 2021
"""
# import os, sys
import numpy as np
from aestheticsModel import simExperiment

def unpackParameters(parameters, n_features=2,
                     n_base_stims=7, n_morphs=5, fixParameters=None,
                     stimSpacing = 'linear'):
    """
    Read in a list with all parameter values in sorted order and returns them
    as named model parameters for passing them to further functions

    Parameters
    ----------
    parameters : list
        Sorted list of parameter values.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    fixParameters : dict, optional
        Dictionary containing values for any fixed (structural) model
        parameters. The dict may contain one or several of 'w_V', 'w_r', 'bias'.
        The default is None.
    stimSpacing : str, optional
        Either 'linear' or 'quadratic'. Determines how the morphs are spaced
        in between source images. 'quadratic' uses np.logspace to create 0.33
        and 0.67 morphs that are closer to the source images than 0.5 morphs.
        The default is 'linear'.

    Returns
    -------
    alpha : float
        Value for the learning rate parameter alpha.
    w_V : float
        Value for the weight of delta-V.
    w_r : float
        Value for the weight of immediate reward r.
    bias : float
        Value for the added bias parameter w_0.
    mu_0 : list of int
        List f length n_features containing means of the system state.
    cov_state : array_like
        Covariance matrix of the agent's system state, assumed to be spherical.
    mu_true : list of int
        List f length n_features containing means of the
        expected true distribution.
    cov_true : array_like
        Covariance matrix of the agent's expected true districution,
        assumed to be spherical.
    stims : array_like
        Array of n_features-dimensional representations of each stimulus.

    """

    nStruct= 4 # basic assumption is that we fit all 4 structural parameters

    # get structural model parameters
    alpha = parameters[0]
    # for remaining parameters, check if any is set
    if fixParameters:
        if 'w_V' in fixParameters:
            w_V = fixParameters['w_V']
            nStruct = nStruct - 1
        else:
            w_V = parameters[1]
        if 'w_r' in fixParameters:
            w_r = fixParameters['w_r']
            nStruct = nStruct - 1
        else:
            w_r = parameters[nStruct-2]
        if 'bias' in fixParameters:
            bias = fixParameters['bias']
            nStruct = nStruct - 1
        else:
            bias = parameters[nStruct-1]
    else:
        w_V = parameters[1]
        w_r = parameters[2]
        bias = parameters[3]

    # agent
    mu_0 = parameters[nStruct:nStruct+n_features]
    cov_state = np.eye(n_features)*parameters[nStruct+n_features]

    mu_true = parameters[nStruct+1+n_features:nStruct+1+n_features*2]
    cov_true = np.eye(n_features)*parameters[nStruct+1+n_features*2]

    # stimuli
    base_stims = []
    start = nStruct + n_features*2 + 2 # number of parameters already assigned
    for b in range(n_base_stims):
        base_stims.append(parameters[start:start+n_features])
        start += n_features

    stims = np.array(base_stims)
    pairs = [[0,2],[0,4],[0,5],[0,6],
             [1,2],[1,3],[1,4],[1,5],[1,6],
             [4,2],[4,3],
             [5,3],[5,4],
             [6,3],[6,4],[6,5]]

    for stimPair in pairs:
        s1 = base_stims[stimPair[0]]
        s2 = base_stims[stimPair[1]]
        if stimSpacing=='linear':
            add_stims = np.linspace(s1, s2, n_morphs)
        elif stimSpacing=='quadratic':
            add_left = np.geomspace(s1, s2, n_morphs)[:3]
            add_right = np.geomspace(s2, s1, n_morphs)[3:]
            add_stims = np.concatenate((add_left, add_right))
        else:
            raise ValueError(("Unknown stimulus spacing. Must be one of:"+
                              "linear; quadratic"))
        add_stims = add_stims[1:-1] # don't keep the base images again
        stims = np.concatenate((stims, add_stims))

    return alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true, stims

def predict(parameters, data, n_features=2, n_base_stims=7, n_morphs=5,
            pre_expose=False, exposure_data=None, fixParameters=None,
            stimSpacing='linear'):
    """
    Predict ratings for all stimuli as indexed by data.imageInd given their
    viewing time recorded as data.rt

    Parameters
    ----------
    parameters : list of float
        Sorted list of parameter values.
    data : pandas df
        pd.DataFrame containing the data to be predicted.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    fixParameters : dict, optional
        Dictionary containing values for any fixed (structural) model
        parameters. The dict may contain one or several of 'w_V', 'w_r', 'bias'.
        The default is None.
    pre_expose : boolean, optional
        Whether or not the agent is exposed to exposure_stims before start
        of predictions. The default is False.
    exposure_data : pandas df, optional
        pd.DataFrame containing data from the free viewing phase. Needs to
        contain image indices and viewing times. Only required if pre_expose=True
        The default is None.
    stimSpacing : str, optional
        Either 'linear' or 'quadratic'. Determines how the morphs are spaced
        in between source images. 'quadratic' uses np.logspace to create 0.33
        and 0.67 morphs that are closer to the source images than 0.5 morphs.
        The default is 'linear'.

    Returns
    -------
    predictions : list of float
        Ratings as predicted by the model specified by parameters.

    """
    if not fixParameters:
        (alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true,
         raw_stims) = unpackParameters(parameters, n_features, n_base_stims,
                                       n_morphs, stimSpacing=stimSpacing)
    else:
         (alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true,
         raw_stims) = unpackParameters(parameters, n_features, n_base_stims,
                                       n_morphs, fixParameters, stimSpacing)

    if pre_expose:
        exposure_stims = raw_stims[exposure_data.imageInd.values,:]
        exposure_stim_durs = np.round(exposure_data.viewTime.values)
        for ii in range(len(exposure_stim_durs)):
            stim = exposure_stims[ii,:]
            dur = exposure_stim_durs[ii]
            # since the attention check introduces NaN values, it's important
            # to check for these and NOT include them
            if not np.isnan(dur):
                mu_0 = simExperiment.simulate_practice_trials(mu_0, cov_state,
                                                          alpha, stim,
                                                          1, dur)

    stims = raw_stims[data.imageInd.values,:]
    stim_dur = np.round(data.rt.values)
    predictions = simExperiment.predict_ratings(mu_0, cov_state, mu_true,
                                            cov_true, alpha,
                                            w_r, w_V, stims, stim_dur, bias)
    return predictions

def predictGroup(parameters, data, n_features=2, n_base_stims=7, n_morphs=5,
                 fixParameters=None, individual_pT=False):
    """
    Predict ratings of an entire group of participants at once assumind stable
    parameters across subjects while letting state parameters vary.

    Parameters
    ----------
    parameters : list of float
        Sorted list of parameter values.
    data : pandas df
        pd.DataFrame containing the data to be predicted.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    fixParameters : dict, optional
        Dictionary containing values for any fixed (structural) model
        parameters. The dict may contain one or several of 'w_V', 'w_r', 'bias'.
        The default is None.
    individual_pT : bool, optional
        Whether parameters describing the expected true distribution are fit
        specified separately for each participant. The default is False.

    Returns
    -------
    predictions : list of float
        Ratings as predicted by the model specified by parameters.

    """
    nParticipants = len(np.unique(data.subj))
    stimParamCount = n_base_stims*n_features
    predictions = []

    for ii in range(nParticipants):
        peep = np.unique(data.subj)[ii]
        stateParamStart = 4+(n_features+1)*ii

        if individual_pT:
            expTrueParamStart = nParticipants*(n_features+1)+stateParamStart
        else:
            expTrueParamStart = nParticipants*(n_features+1)+4

        stateParamRange =  np.arange(stateParamStart,
                                     stateParamStart+n_features+1)
        expTrueParamRange = np.arange(expTrueParamStart,
                                      expTrueParamStart+n_features+1)

        # select data and parameters for this participant
        thisData = data[data.subj==peep]
        structParam = parameters[:4]
        stateParam = parameters[stateParamRange]
        expTrueParam = parameters[expTrueParamRange]
        stimParam = parameters[-stimParamCount:]
        thisParameters = [*structParam, *stateParam, *expTrueParam,
                          *stimParam]

        pred = predict(thisParameters, thisData, n_features, n_base_stims,
                       n_morphs, fixParameters)
        predictions.append(pred)
    predictions = [item for sublist in predictions for item in sublist]
    return predictions

def predict_viewTimes(parameters, data, n_features=2,
                      n_base_stims=7, n_morphs=5,
                      t_min = 1, t_max = 1e3,
                      thresholdType = 'directComparison',
                      fixThreshold=0.5, beta=1,
                      add_valueNoise=False, noiseSeed=42):
    """
    Predict ratings for all stimuli as indexed by data.imageInd given their
    viewing time recorded as data.rt

    Parameters
    ----------
    parameters : list of float
        Sorted list of parameter values.
    data : pandas df
        pd.DataFrame containing the data to be predicted.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    t_min : int, optional
        Number of stimuli per morphed pair. The default is 1.
    t_max : int, optional
        Number of stimuli per morphed pair. The default is 1e3.
    thresholdType : str, optional
        Number of stimuli per morphed pair. The default is 'directComparison'.
    fixThreshold : float, optional
        Number of stimuli per morphed pair. The default is 0.5.
    beta : float, optional
        temperature parameter for computing threshold if thresholdType
        is 'discountedMean'.
    add_valueNoise : bool, optional
        Whether or not to add noise (range: +- 0.5) to the threshold.
        The default is False.
    noiseSeed : int, optional
        Seed for the random number generator that creates noise if added.
        The default is 42.

    Raises
    ------
    ValueError
        Raises ValueError if thresholdType is unspecified.

    Returns
    -------
    predictions : list of int
        List of predicted viewing times for each stimulus in data.

    """

    alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true, raw_stims = unpackParameters(parameters, n_features, n_base_stims, n_morphs)
    stims = raw_stims[data.imageInd.values,:]

    predictions = []
    A_t_list = []
    new_mu = mu_0.copy()
    for trial in range(len(stims)-1):
        stim = stims[trial,:]
        next_stim = stims[trial+1,:]

        if thresholdType=='directComparison':
            threshold = simExperiment.calc_predictions(new_mu, cov_state,
                                                       mu_true, cov_true, alpha,
                                                       next_stim, 0, w_r, w_V,
                                                       bias)
        elif thresholdType=='fix':
            threshold = fixThreshold # this should be a parameter in the future
        elif thresholdType=='grandMean':
            A_t = simExperiment.calc_predictions(new_mu, cov_state,
                                                       mu_true, cov_true, alpha,
                                                       next_stim, 0, w_r, w_V,
                                                       bias)
            A_t_list.append(A_t)
            threshold = np.nanmean(A_t_list)
        elif thresholdType=='discountedMean':
            A_t = simExperiment.calc_predictions(new_mu, cov_state,
                                                       mu_true, cov_true, alpha,
                                                       next_stim, 0, w_r, w_V,
                                                       bias)
            A_t_list.append(A_t)
            discountWeights = list(reversed([beta**t for t in range(trial+1)]))
            threshold = np.average(A_t_list, weights=discountWeights)
        else:
            raise ValueError(("Unknown threshold type. Must be one of:"+
                              "directComparison; fix; grandMean; discountedMean"))
        if add_valueNoise:
            np.random.seed(noiseSeed)
            threshold += np.random.rand(1)-0.5 #center noise around 0

        viewTime, new_mu = simExperiment.get_view_time(new_mu, cov_state,
                                         mu_true, cov_true,
                                         alpha, stim, threshold=threshold,
                                         w_r=w_r, w_V=w_V, bias=bias,
                                         min_t=t_min, max_t=t_max,
                                         return_mu=True)

        predictions.append(viewTime)
    return predictions