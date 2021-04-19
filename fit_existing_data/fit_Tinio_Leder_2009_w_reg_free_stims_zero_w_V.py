# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 2020; major update Tue Feb 2 2021

@author: abrielmann

Fits the main parameters of the model of aesthetic value developed by
Aenne Brielmann and Peter Dayan
to the results reported by Tinio & Leder (2009)
"""
# import standard python packages needed
import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import nnls
from matplotlib import pyplot as plt
# import costum functions
os.chdir('..')
home_dir = os.getcwd()
sys.path.append((home_dir + "/python_packages"))
from aestheticsModel import simExperiment

# set some general parameters for these simulations
plot = False # generate plot of results vs data?
write_csv = True # write results into csv file?

bounds = ((-np.inf, np.inf), (-np.inf, np.inf), # state mus
          (0, np.inf), # state var
          (-np.inf, np.inf), # state rho
          (-np.inf, np.inf), (-np.inf, np.inf), # mus for p_true
          (0, np.inf), # p_true var
          (-np.inf, np.inf), # true rho
          (-np.inf, np.inf), (-np.inf, np.inf), #deviation of complex/symmetric stims
          (0, 1)) # alpha

# %% enter results reported by Tinio & Leder (2009)
means_exp1 = [4.62, 3.86, 3.31, 2.38]

means_exp2_CoSy = [5.02, 3.96, 3.29, 2.41]
means_exp2_SiSy = [5.57, 3.84, 3.84, 1.99]
means_exp2_CoNs = [4.76, 4.67, 3.11, 2.81]
means_exp2_SiNs = [4.39, 3.71, 2.97, 2.31]

means_exp3_CoSy = [4.95, 3.67, 3.5, 2.44]
means_exp3_SiSy = [4.18, 3.45, 3.05, 2.3]
means_exp3_CoNs = [5.16, 3.95, 3.73, 2.1]
means_exp3_SiNs = [4.75, 3.75, 3.23, 2]

all_reported_means = np.array([means_exp1,
                      means_exp2_CoSy, means_exp2_SiSy,
                      means_exp2_CoNs, means_exp2_SiNs,
                      means_exp3_CoSy, means_exp3_SiSy,
                      means_exp3_CoNs, means_exp3_SiNs]).flatten()

# %% define a function that returns predictions for given sequence of stimuli
# and possibly familiarization trials
def predict_avg_response(mu, cov, mu_true, cov_true, alpha, stims,
                         sensory_weight=1, w_V=1,
                         bias=0, n_famil_trials=0, famil_stim=[]):
    # create copies of initial mu and cov as basis for true environment dist
    new_mu = mu.copy()

    # if familiarization stimuli are provided, run through familiarization first
    if n_famil_trials != 0:
        new_mu = simExperiment.simulate_practice_trials(mu, cov, alpha,
                                                           famil_stim,
                                                           n_stims=n_famil_trials,
                                                           stim_dur=1)
    # get predicted responses
    r_t_list = []
    dV_list = []
    for stim in stims:
        _, r_t, dV = simExperiment.calc_predictions(new_mu, cov, mu_true, cov_true, alpha,
                                         stim, return_r_t=True, return_dV=True)
        r_t_list.append(r_t)
        dV_list.append(dV)

    return np.array(r_t_list), np.array(dV_list)

# %% for after fitting, we also want a fn that returns predicted avgs
def predict_experiment_data(parameters, data):

    mu_state = parameters[:2]
    var_state = parameters[2]**2
    rho_state = var_state**2 * parameters[3]
    cov_state = [[var_state, rho_state],
                           [rho_state, var_state]]
    # repeat, for true distribution
    mu_true = parameters[4:6]
    var_true = parameters[6]**2
    rho_true = var_true**2 * parameters[7]
    cov_true = [[var_true, rho_true],
                           [rho_true, var_true]]
    alpha = parameters[-1]

    # Define the experimental stimulus set
    # For fitting, we only assume means of the distributions for each condition
    CoSy_stim = [0+parameters[-3], 0+parameters[-2]]
    SiSy_stim = [0, 0+parameters[-2]]
    CoNs_stim = [0+parameters[-3], 0]
    SiNs_stim = [0, 0]
    stims = np.vstack((CoSy_stim, SiSy_stim, CoNs_stim, SiNs_stim))

    r_exp1, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims)
    r_exp2a, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=320, famil_stim=stims[0,:])
    r_exp2b, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=320, famil_stim=stims[1,:])
    r_exp2c, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=320, famil_stim=stims[2,:])
    r_exp2d, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=320, famil_stim=stims[3,:])
    r_exp3a, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=80, famil_stim=stims[0,:])
    r_exp3b, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=80, famil_stim=stims[1,:])
    r_exp3c, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=80, famil_stim=stims[2,:])
    r_exp3d, _ = predict_avg_response(mu_state, cov_state, mu_true, cov_true, alpha,
                                     stims,
                                     n_famil_trials=80, famil_stim=stims[3,:])

    pred_r = np.array([r_exp1,
                   r_exp2a, r_exp2b, r_exp2c, r_exp2d,
                   r_exp3a, r_exp3b, r_exp3c, r_exp3d]).flatten()

    try:
        # lstsq with non-negativity constraint
        weights = nnls(np.array([pred_r, np.ones(len(pred_r))]).T,
                               data)[0]
        predictions = weights[0]*pred_r + weights[1]
    except:
        weights = [1,0]
        predictions = pred_r

    return predictions, weights

#%% Define the cost function
def cost_fn(parameters, data):
    predictions, _ = predict_experiment_data(parameters, data)
    cost = np.mean(np.abs(data - predictions))
    return cost

#%% minimization
# a few of these will get stuck and return nan after maxiter has been reached
for seed in range(1000):
    # set randomization seed for reproducibility
    np.random.seed(seed)
    parameters = np.random.rand(11)
    # scale the starting point for alpha
    parameters[-1] = parameters[-1]/1e3
    additional_arguments = tuple((all_reported_means,))

    res = minimize(cost_fn, parameters, args=additional_arguments,
                    method='SLSQP',
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 1e3, 'ftol': 1e-07})
    # print(res)
    x_res = res.x

    # get predictions
    predictions, weights = predict_experiment_data(x_res, all_reported_means)
    mu_state_1 = x_res[0]
    mu_state_2 = x_res[1]
    var_state = x_res[2]**2
    rho_state = var_state**2*x_res[3]
    # repeat, for true distribution
    mu_true_1 = x_res[4]
    mu_true_2 = x_res[5]
    var_true = x_res[6]**2
    rho_true = var_true**2 * x_res[7]
    complex_add = x_res[-3]
    symmetry_add = x_res[-2]
    alpha = x_res[-1]
    w_r = weights[0]
    w_V = 0
    bias = weights[1]

    print(('Iteration ' + str(seed) + ' done.'))
    # automatically append simulation results to the .csv
    if write_csv:
        rmse = res.fun
        success = res.success
        pred_list = predictions.tolist()
        res_list = [seed, rmse, success, mu_state_1, mu_state_2, var_state, rho_state,
                    mu_true_1, mu_true_2, var_true, rho_true, w_r, w_V, alpha, bias,
                    complex_add, symmetry_add] + pred_list
        myCsvRow = ",".join(map(str, res_list))
        myCsvRow = myCsvRow + "\n"
        with open((home_dir +
                    '/simulate_existing_experiments/'
                    + 'it_results_Tinio_2009_w_nnls_reg_free_stims_zero_w_V.csv'),'a') as fd:
            fd.write(myCsvRow)

    # and plot them vs. data
    if plot:
        sub_exp_names = ['Baseline', 'Long familiarization CoSy', 'Long familiarization SiSy',
                         'Long familiarization CoNs', 'Long familiarization SiNs',
                         'Short familiarization CoSy', 'Short familiarization SiSy',
                         'Short familiarization CoNs', 'Short familiarization SiNs']
        # for this, SDs will be good to display,too
        sds_reported = [1.23, 0.91, 1.01, 0.98,
                        1.37, 0.79, 1.11, 0.99,
                        0.87, 1.03, 0.72, 0.48,
                        0.7, 0.57, 0.49, 0.88,
                        1.18, 0.52, 0.64, 0.73,
                        1.1, 0.86, 1.06, 1.08,
                        1.33, 1.28, 1.27, 1.06,
                        0.51, 0.93, 1.04, 0.81,
                        1.34, 1.07, 0.91, 0.74]

        # set up the big figure
        fig, axs = plt.subplots(3,3)
        axs.ravel()
        ax_idxs = ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2))
        # plot results from each experiment in a separate subplot
        for sub_exp in range(len(sub_exp_names)):
            data = all_reported_means[sub_exp*4:((sub_exp+1)*4)]
            data_sd = sds_reported[sub_exp*4:((sub_exp+1)*4)]
            pred = predictions[sub_exp*4:((sub_exp+1)*4)]

            axs[ax_idxs[sub_exp]].set_title(sub_exp_names[sub_exp])
            color = 'tab:gray'
            axs[ax_idxs[sub_exp]].set_xlabel('Stimulus category')
            axs[ax_idxs[sub_exp]].set_ylabel('Data', color=color)
            axs[ax_idxs[sub_exp]].errorbar(np.arange(0,4), data, data_sd,
                                           fmt='o', color=color)
            # cosmetics
            axs[ax_idxs[sub_exp]].tick_params(axis='y', labelcolor=color)
            axs[ax_idxs[sub_exp]].spines['top'].set_visible(False)
            axs[ax_idxs[sub_exp]].set_xticks(np.arange(0,4))
            axs[ax_idxs[sub_exp]].set_xticklabels(['CoSy', 'SiSy', 'CoNs', 'SiNs'])

            ax2 = axs[ax_idxs[sub_exp]].twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Model', color=color)  # we already handled the x-label with ax1
            ax2.plot(pred, 'o', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            # cosmetics
            ax2.spines['top'].set_visible(False)
            ax2.set_yticks([])
            # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        fig.tight_layout(pad=.25)  # otherwise the right y-label is slightly clipped
        fig.set_figheight(7)
        fig.set_figwidth(8)
        plt.show()
        plt.close()