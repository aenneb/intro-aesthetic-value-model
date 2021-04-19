# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:55:31 2021

@author: abrielmann
"""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import and update global figure settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'legend.title_fontsize': 12,
          'legend.borderpad': 0,
          'figure.figsize': (10,6),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         'lines.linewidth': 2,
         'image.cmap': 'gray',
         'savefig.dpi': 300}
pylab.rcParams.update(params)

# import costum functions
os.chdir('..')
home_dir = os.getcwd()
sys.path.append((home_dir + "/python_packages"))

# %% define a few helper functions we need to process the data and plot
def exponential(C, k, reps):
    y = [(C * ( 1 - np.exp(-k*t))) for t in reps]
    return np.array(y)

def get_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extend_fit_results(fit_results_raw):
    fit_results = fit_results_raw.loc[fit_results_raw.success==True]
    fit_results = fit_results.loc[fit_results.rmse<1]
    best_fit = fit_results.loc[fit_results['rmse']==np.min(fit_results['rmse'])]

    # in case of the modified fits, we get several equally good fits, pick one
    if len(best_fit)>1:
        best_fit = best_fit.iloc[0,:]
        best_fit = best_fit.to_frame().T
        best_fit = best_fit.astype(best_fit.infer_objects().dtypes)

    return fit_results, best_fit

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

def get_dist_plot_data(fit, xmin=-10, ymin = -10, xmax=100, ymax=100):
    N = 50
    X = np.linspace(xmin, xmax, N)
    Y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    mu_X = np.array([0,0])
    cov_X = np.array([[fit.var_state.values.squeeze(), 0],
             [0, fit.var_state.values.squeeze()]]).squeeze()
    mu_true = np.array([fit.mu_true_1.values, fit.mu_true_2.values]).squeeze()
    cov_true = np.array([[fit.var_true.values.squeeze(), 0],
                [0, fit.var_true.values.squeeze()]]).squeeze()
    Z_X = multivariate_gaussian(pos, mu_X, cov_X)
    Z_true = multivariate_gaussian(pos, mu_true, cov_true)

    return X, Y, Z_X, Z_true

def plot_mere_exposure(data, predictions, best_fit, X, Y, Z_X, Z_true,
                       ax0, ax1, panel_label = 'A', legends=False):
    #  data vs. pred
    ax0.plot(np.arange(1,51), predictions, '--',
             label='model', linewidth=3)
    ax0.plot(np.arange(1,51), data, label='data', linewidth=2)
    ax0.set_xticks(np.arange(0,51,10))
    ax0.set_xticklabels(np.arange(0,51,10))
    if legends:
        ax0.legend()
    ax0.set_ylabel('Average rating / predictions')
    ax0.set_xlabel('Repetition #')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    # add a title that shows information about structural parameters
    s1 = (r'$\alpha = $' + str(np.round(best_fit.alpha.values[0],4))
                  + r'; $w^r = $' + str(np.round(best_fit.w_r.values[0],2))
                  + r'; $w^V = $' + str(np.round(best_fit.w_V.values[0],2)))
    s2 = '; $RMSE = $' + str(np.round(best_fit.rmse.values[0],5))
    ax0.set_title((s1 + s2),
                  fontdict={'fontsize': 10, 'horizontalalignment': 'left'})
    # add panel label
    ax0.text(-.5, 1.1, panel_label, transform=ax0.transAxes,
      fontsize=12, fontweight='bold', va='top', ha='right')

    # plot best fit system state and true distributiosn
    ax1.contour(X, Y, Z_X, cmap='Blues') #zdir='z', offset=0,
    if legends:
        ax1.text(0, 10, '$X(0)$',
                 fontdict=({'color':  'royalblue', 'size': 14}))
    ax1.contour(X, Y, Z_true, cmap='Greens') #zdir='z', offset=0,
    if legends:
        ax1.text(best_fit.mu_true_1-20, best_fit.mu_true_2, '$p^T$',
                 fontdict=({'color':  'forestgreen', 'size': 14}))
    ax1.plot(best_fit.stim_mu_1.values, best_fit.stim_mu_2.values, 'or',
            markersize=5, label='stimulus')
    if legends:
        ax1.legend(loc='upper right')
    ax1.set_xticks((np.min(X),np.max(X)))
    ax1.set_yticks((np.min(Y),np.max(Y)))
    ax1.grid(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

#%% get fit results
fit_results_raw = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Montoya_2017.csv', index_col=False)
fit_results_raw_zero_w_r = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Montoya_2017_zero_w_r.csv', index_col=False)
fit_results_raw_zero_w_V = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Montoya_2017_zero_w_V.csv', index_col=False)
fit_results_raw_equal_weights = pd.read_csv(home_dir + '/results'
                          + '/fit_results_Montoya_2017_equal_weights.csv', index_col=False)

fit_results, best_fit = extend_fit_results(fit_results_raw)
fit_results_zero_w_r, best_fit_zero_w_r = extend_fit_results(fit_results_raw_zero_w_r)
fit_results_zero_w_V, best_fit_zero_w_V = extend_fit_results(fit_results_raw_zero_w_V)
fit_results_equal_weights, best_fit_equal_weights = extend_fit_results(fit_results_raw_equal_weights)

#%% get data and predictions
reps = np.arange(1,51)
# inv u
meta_intercept = .66051
meta_slope = 1.73e-03
meta_quadratic = -2.5e-05
data = np.polyval([meta_quadratic, meta_slope, meta_intercept], reps)
predictions = best_fit.iloc[:,10:60].values.flatten()

predictions_zero_w_r = best_fit_zero_w_r.iloc[:,10:60].values.flatten()
predictions_zero_w_V = best_fit_zero_w_V.iloc[:,10:60].values.flatten()
predictions_equal_weights = best_fit_equal_weights.iloc[:,10:60].values.flatten()

#%%  PLOT
# use gridspec to define layout
fig = plt.figure()
gs = fig.add_gridspec(2,4)

# inverted U
ax0 = fig.add_subplot(gs[0, 0]) # data vs. prediction
ax1 = fig.add_subplot(gs[0, 1]) # distribtuions
# U-shape
ax2 = fig.add_subplot(gs[0, 2]) # data vs. prediction
ax3 = fig.add_subplot(gs[0, 3]) # distribtuions
# increase
ax4 = fig.add_subplot(gs[1, 0]) # data vs. prediction
ax5 = fig.add_subplot(gs[1, 1]) # distribtuions
# decrease
ax6 = fig.add_subplot(gs[1, 2]) # data vs. prediction
ax7 = fig.add_subplot(gs[1, 3]) # distribtuions

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit,
                                       xmin=-10, ymin = -10, xmax=60, ymax=60)
plot_mere_exposure(data, predictions, best_fit,
                   X, Y, Z_X, Z_true, ax0, ax1, legends=True)

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit_zero_w_r,
                                       xmin=-10, ymin = -10, xmax=20, ymax=20)
plot_mere_exposure(data, predictions_zero_w_r, best_fit_zero_w_r,
                   X, Y, Z_X, Z_true, ax2, ax3, panel_label = 'B')

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit_zero_w_V,
                                        xmin=-10, ymin = -10, xmax=20, ymax=20)
plot_mere_exposure(data, predictions_zero_w_V, best_fit_zero_w_V,
                   X, Y, Z_X, Z_true, ax4, ax5, panel_label = 'C')

X, Y, Z_X, Z_true = get_dist_plot_data(best_fit_equal_weights,
                                        xmin=-10, ymin = -10, xmax=20, ymax=20)
plot_mere_exposure(data, predictions_equal_weights, best_fit_equal_weights,
                   X, Y, Z_X, Z_true, ax6, ax7, panel_label = 'D')

fig.tight_layout(pad=.25)
plt.savefig('summary_fig_mere_exposure_v9_lesions.svg')
