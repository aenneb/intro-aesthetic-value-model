# -*- coding: utf-8 -*-
"""
Created on Tue Feb 2 2021

@author: abrielmann
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import and update global figure settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'legend.title_fontsize': 12,
          'legend.borderpad': 0,
          'figure.figsize': (8, 8),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         'lines.linewidth': 2,
         'image.cmap': 'gray',
         'savefig.dpi': 300}
pylab.rcParams.update(params)

# home dir
os.chdir('..')
home_dir = os.getcwd()

# %% experimental data

# experimental results
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
sds_reported = [1.23, 0.91, 1.01, 0.98,
                        1.37, 0.79, 1.11, 0.99,
                        0.87, 1.03, 0.72, 0.48,
                        0.7, 0.57, 0.49, 0.88,
                        1.18, 0.52, 0.64, 0.73,
                        1.1, 0.86, 1.06, 1.08,
                        1.33, 1.28, 1.27, 1.06,
                        0.51, 0.93, 1.04, 0.81,
                        1.34, 1.07, 0.91, 0.74]

# %% define functions that extend fits, plot, etc.
def extend_fit_results(fit_results_raw):
    # discard non-converged ones
    fit_results = fit_results_raw.loc[fit_results_raw.success==True]
    fit_results = fit_results[fit_results.alpha<1]

    # extract the best fit
    best_fit = fit_results.loc[fit_results['rmse']==np.min(fit_results['rmse'])]
    # in case of the modified fits, we get several equally good fits, pick one
    if len(best_fit)>1:
        best_fit = best_fit.iloc[0,:]
        best_fit = best_fit.to_frame().T
        best_fit = best_fit.astype(best_fit.infer_objects().dtypes)

    best_predictions = best_fit.iloc[:,17:(17+36)].values.flatten()

    return fit_results, best_fit, best_predictions

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def get_dist_plot_data(fit, xmin=-10, ymin = -10, xmax=10, ymax=10):
    N = 25
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    mu_X = np.array([fit.mu_state_1.values, fit.mu_state_2.values]).squeeze()
    cov_X = np.array([[fit.var_state.values.squeeze(), fit.rho_state.values.squeeze()],
             [fit.rho_state.values.squeeze(), fit.var_state.values.squeeze()]]).squeeze()
    mu_true = np.array([fit.mu_true_1.values, fit.mu_true_2.values]).squeeze()
    cov_true = np.array([[fit.var_true.values.squeeze(), fit.rho_true.values.squeeze()],
                [fit.rho_true.values.squeeze(), fit.var_true.values.squeeze()]]).squeeze()
    Z_X = multivariate_gaussian(pos, mu_X, cov_X)
    Z_true = multivariate_gaussian(pos, mu_true, cov_true)

    return X, Y, Z_X, Z_true

def create_plots(all_reported_means, best_predictions, fit, X, Y, Z_X, Z_true,
                 ax0,ax1, ax1b, panel_label='A', min_zoom=0, max_zoom=1, 
                 legend=False):
    # simple things first: plot correlation between data and results
    ax0.plot(all_reported_means, best_predictions, '.k')
    ax0.set_xlim((1,6))
    ax0.set_ylim((1,6))
    ax0.set_xlabel('Data')
    ax0.set_ylabel('Predictions')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
     # add panel label
    ax0.text(-.25, 1.1, panel_label, transform=ax0.transAxes,
      fontsize=12, fontweight='bold', va='top', ha='right')

    # 2d distributions for best fit
    ax1.plot(fit.complex_add, fit.symmetry_add, 'or', markersize=8,
             label='CoSy', alpha=.8)
    ax1.plot(0, fit.symmetry_add, 'ok', markersize=8, label='SiSy', alpha=.8)
    ax1.plot(fit.complex_add, 0, 'xr', markersize=8, label='CoNs', alpha=.8)
    ax1.plot(0, 0, 'xk', markersize=8, label='SiNs', alpha=.8)
    if legend:
        ax1.legend(loc='lower right', ncol=2)
    ax1.contour(X, Y, Z_X, cmap='Blues') 
    ax1.contour(X, Y, Z_true, cmap='Greens') 
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_xlim((min_zoom,max_zoom))
    ax1.set_xlabel('Complexity')
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_ylim((min_zoom,max_zoom))
    ax1.set_ylabel('Symmetry')
    ax1.grid(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # add a title w information about parameters
    ax1.set_title((r'$\alpha = $' + str(np.round(fit.alpha.values[0],4))
                  + r'; $w^r = $' + str(np.round(fit.w_r.values[0],2))
                  + r'; $w^V = $' + str(np.round(fit.w_V.values[0],0))
                  + '; $RMSE = $' + str(np.round(fit.rmse.values[0],3))),
                  fontdict={'horizontalalignment': 'center'})

    # X(0) and pT
    shift = (np.max(X) - np.min(X)) / 8
    ax1b.contour(X, Y, Z_X, cmap='Blues') #zdir='z', offset=0,
    ax1b.text(fit.mu_state_1-shift, fit.mu_state_2+shift, '$X(0)$',
                fontdict=({'color':  'royalblue', 'size': 14}))
    ax1b.contour(X, Y, Z_true, cmap='Greens') #zdir='z', offset=0,
    ax1b.text(fit.mu_true_1+shift, fit.mu_true_2+shift, '$p^T$',
              fontdict=({'color':  'forestgreen', 'size': 14}))
    # add SiNs for reference
    ax1b.plot(0,0,'xk')
    ax1b.text(.5, -1, 'SiNs', fontdict=({'size': 14}))
    ax1b.set_xticks([])
    ax1b.set_xticklabels([])
    ax1b.set_xlim((-5,5))
    ax1b.set_xlabel('Complexity')
    ax1b.set_yticks([])
    ax1b.set_yticklabels([])
    ax1b.set_ylim((-5,5))
    ax1b.set_ylabel('Symmetry')
    ax1b.grid(False)
    ax1b.spines['right'].set_visible(False)
    ax1b.spines['top'].set_visible(False)

#%% get fit results
fit_results_raw = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Tinio_2009_w_nnls_reg_free_stims.csv',
                          index_col=False)
fit_results, best_fit, best_predictions = extend_fit_results(fit_results_raw)

fit_results_raw = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Tinio_2009_w_nnls_reg_free_stims_zero_w_V.csv',
                          index_col=False)
fit_results_zero_w_V, best_fit_zero_w_V, best_predictions_zero_w_V = extend_fit_results(fit_results_raw)

fit_results_raw = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Tinio_2009_w_nnls_reg_free_stims_equal_weights.csv',
                          index_col=False)
fit_results_equal_weights, best_fit_equal_weights, best_predictions_equal_weights = extend_fit_results(fit_results_raw)

#%% PLOT
# use gridspec to define layout
fig = plt.figure()
gs = fig.add_gridspec(3,3)

# overall best fit
ax0 = fig.add_subplot(gs[0, 0]) # data vs. prediction
ax1 = fig.add_subplot(gs[0, 1]) # distribtuion X(0) w stimuli
ax2 = fig.add_subplot(gs[0, 2]) # distribtuion X(0) vs pT

# best fit w_V = 0
ax3 = fig.add_subplot(gs[1, 0]) # data vs. prediction
ax4 = fig.add_subplot(gs[1, 1]) # distribtuion X(0) w stimuli
ax5 = fig.add_subplot(gs[1, 2]) # distribtuion X(0) vs pT

# best fit equal weights
ax6 = fig.add_subplot(gs[2, 0]) # data vs. prediction
ax7 = fig.add_subplot(gs[2, 1]) # distribtuion X(0) w stimuli
ax8 = fig.add_subplot(gs[2, 2]) # distribtuion X(0) vs pT

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit,
                                       xmin=-5, ymin =-5, xmax=5, ymax=5)
create_plots(all_reported_means, best_predictions,
                               best_fit, X, Y, Z_X, Z_true, ax0, ax1, ax2,
                               min_zoom=-0.1, max_zoom=0.1, legend=True)

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit_zero_w_V,
                                       xmin=-5, ymin =-5, xmax=5, ymax=5)
create_plots(all_reported_means, best_predictions_zero_w_V,
             best_fit_zero_w_V, X, Y, Z_X, Z_true, ax3, ax4, ax5, 'B',
                               min_zoom=-1, max_zoom=4)

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit_equal_weights,
                                       xmin=-5, ymin =-5, xmax=5, ymax=5)
create_plots(all_reported_means, best_predictions_equal_weights,
                               best_fit_equal_weights, X, Y, Z_X, Z_true,
                               ax6, ax7, ax8, 'C',
                               min_zoom=-1, max_zoom=1)


fig.tight_layout(pad=1)
plt.savefig('summary_fig_tinio_leder_v9.svg')
