# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23

@author: abrielmann

Fits the main parameters of the model of aesthetic value developed by
Aenne Brielmann and Peter Dayan
to the results reported by Montoya et al. (2017)
"""
# import standard python packages needed
import os
import sys
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# import costum functions
os.chdir('..')
home_dir = os.getcwd()
sys.path.append((home_dir + "/python_packages"))
from aestheticsModel import simExperiment

# set further general parameters
n_features = 2
plot = False # generate plot of results vs data?
write_csv = True
n_iterations = 1000

# bounds depend on number of specified features
bounds = ((-np.inf, np.inf), (-np.inf, np.inf), # differencse between agent and stim mus
          (0, np.inf), # agent var
          (-np.inf, np.inf), (-np.inf, np.inf), # mus for p_true
          (0, np.inf), # p_true var
          (0, np.inf), (0, np.inf), (0, 1), (-np.inf, np.inf)) # w_r, w_V, alpha, bias

reps = np.arange(1, 51) # number of exposures per stimulus

#%% generate data from polynomial fits
meta_intercept = .66051
meta_slope = 1.73e-03
meta_quadratic = -2.5e-05
data = np.polyval([meta_quadratic, meta_slope, meta_intercept], reps)

# %% define a function that returns predictions
def predict_avg_response(parameters, n_features, reps=reps):

    # agent's system state
    mu = np.repeat(0, n_features).astype(float)
    agent_var = np.abs(parameters[n_features])
    cov = np.eye(n_features)*agent_var

    # p_true
    mu_init = parameters[n_features+1:(n_features*2+1)]
    p_true_var = np.abs(parameters[n_features*2+1])
    cov_init = np.eye(n_features)*p_true_var

    # strucutral
    w_r = parameters[-4]
    w_V = parameters[-3]
    alpha = parameters[-2]
    bias = parameters[-1]

    # stimulus
    stim_mu =  parameters[0:n_features]

    # set up variables to track
    A_t_list = []

    # we run through the different numbers of exposure
    for rep in reps:
        # first, run through exposure trials
        this_mu = simExperiment.simulate_practice_trials(mu, cov, alpha,
                                                            stim_mu, n_stims=1,
                                                            stim_dur=rep)
        # then get final rating
        A_t = simExperiment.calc_predictions(this_mu, cov, mu_init, cov_init,
                                                           alpha, stim_mu,
                                                           w_r, w_V, bias)
        A_t_list.append(A_t)

    return A_t_list

#%% Define the cost function
def cost_fn(parameters, n_features, data):
    predictions = predict_avg_response(parameters, n_features)
    cost = np.sqrt(np.mean((predictions-data)**2))*1e3 # scale up to ease minimization
    return cost

# %% loop over several different starting points
for seed in range(n_iterations):
    # set randomization seed for reproducibility
    np.random.seed(seed)

    # minimization
    parameters = np.random.rand(n_features*2+6)
    parameters[-2] = parameters[-2]/1e3 # scale the starting point for alpha
    additional_arguments = tuple((n_features, data))

    res = minimize(cost_fn, parameters, args=additional_arguments,
                    method='SLSQP', 
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 1e3, 'ftol': 1e-07})
    x_res = res.x.tolist()
    success = res.success
    rmse = res.fun/1e3 # scale rmse back to true number

    # get predictions
    predictions = predict_avg_response(x_res, n_features)

    # %% automatically append results to the .csv
    if write_csv:
        res_list = x_res + predictions + [success] + [rmse] + [seed]
        myCsvRow = ",".join(map(str, res_list)) + "\n"
        with open((home_dir
                    + '/simulate_existing_experiments'
                    + '/fit_results_Montoya_2017.csv'),'a') as fd:
            fd.write(myCsvRow)

    # and plot them vs. data
    if plot:
        plt.plot(predictions, label='best predictions')
        plt.plot(data, label='data')
        plt.legend()
        plt.title(r'$w^V = $' + str(np.round(x_res[-3],6)) +
                  r'; $\alpha = $' + str(np.round(x_res[-2],6)))
        plt.show()
        plt.close()

    print('Iteration ' + str(seed) + ' done.')
