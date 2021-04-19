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

#%% Specifying settings for fits
# bounds depend on number of specified features
bounds = ((-np.inf, np.inf), # differencse between agent and stim mus
          (0, np.inf), # agent var
          (-np.inf, np.inf), # mus for p_true
          (0, np.inf), # p_true var
          (0, np.inf), (0, np.inf), (0, 1), (-np.inf, np.inf)) # w_r, w_V, alpha, bias

reps = np.arange(1, 51) # number of exposures per stimulus

plot = False # generate plot of results vs data?
write_csv = True

#%% generate data from polynomial fits
meta_intercept = .66051
meta_slope = 1.73e-03
meta_quadratic = -2.5e-05
data = np.polyval([meta_quadratic, meta_slope, meta_intercept], reps)

# %% define a function that returns predictions
def predict_avg_response(parameters, reps=reps):

    # stimulus
    stim_mu =  parameters[0]
    # agent's system state
    mu_state = np.array(0)
    var_state = parameters[1]

    # p_true
    mu_true = parameters[2]
    var_true = parameters[3]

    # strucutral
    w_r = parameters[4]
    w_V = parameters[5]
    alpha = parameters[6]
    bias = parameters[7]

    # set up variables to track
    A_t_list = []

    # we run through the different numbers of exposure
    for rep in reps:
        # first, run through exposure trials
        this_mu = simExperiment.simulate_practice_trials(mu_state, var_state, alpha,
                                                            stim_mu, n_stims=1,
                                                            stim_dur=rep)
        # then get final rating
        A_t = simExperiment.calc_predictions(this_mu, var_state, mu_true, var_true,
                                                           alpha, stim_mu,
                                                           w_r, w_V, bias)
        A_t_list.append(A_t)

    return A_t_list

#%% Define the cost function
def cost_fn(parameters, data):
    predictions = predict_avg_response(parameters)
    cost = np.sqrt(np.mean((predictions-data)**2))*1e3 # scale up to ease minimization
    # print(cost)
    return cost

# %% loop over several different starting points
for seed in range(1000):
    # set randomization seed for reproducibility
    np.random.seed(seed)

    # minimization
    parameters = np.random.rand(8)
    parameters[-2] = parameters[-2]/1e3 # scale the starting point for alpha
    additional_arguments = tuple((data,))

    res = minimize(cost_fn, parameters, args=additional_arguments,
                    method='SLSQP',
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 1e3, 'ftol': 1e-07})
    # print(res)
    x_res = res.x.tolist()
    success = res.success
    rmse = res.fun/1e3 # scale rmse bacck to true number

    # get predictions
    predictions = predict_avg_response(x_res)

    # %% automatically append results to the .csv
    if write_csv:
        res_list = x_res + predictions + [success] + [rmse] + [seed]
        myCsvRow = ",".join(map(str, res_list)) + "\n"
        with open((home_dir
                    + '/simulate_existing_experiments'
                    + '/fit_results_Montoya_2017_1d.csv'),'a') as fd:
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
