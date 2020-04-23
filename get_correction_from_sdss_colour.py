import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import george as g


#  ---------------------------------------------------------
#/-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\#
#|=-=-=-=-=-=-=-=-=-=-=-=- Variables -=-=-=-=-=-=-=-=-=-=-=-=|#
#\-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-/#
#  ---------------------------------------------------------

# Which standard star do I plot on the correction tracks?
std_name = 'SA 114 548'
# std_name = 'BD-21 0910'
# std_name = 'BD+17 4708'
# std_name = 'BD+33 2642'

# std_name = 'None'

# What telescope am I correcting on?
telescope = 'ntt'
instrument = 'ucam'
# regular/super
filt = 'super'

# Am I a main sequence (pickles)? Or a white dwarf (koester)?
source = 'pickles'

# Error to add to the y-axis. The Smith values have errors <0.005 mags,
# so subtracted their max error is 0.010. This will do as a rule of thumb for the Pickles data
err = 0.00

# Which combinations do we want to plot? These must match!
x_combinations = ['u-g', 'g-r', 'r-i', 'i-z']
std_colours = ["u'-g'", "g'-r'", "r'-i'", "i'-z'"]

# x_combinations = ['g-r']
# std_colours = ["g'-r'"]

figsize = None #(3.5, 6)
tick_right = False
publishable = False

#  ---------------------------------------------------------
#/-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\#
#|=-=-=-=-=-=-=- The rest should handle itself -=-=-=-=-=-=-=|#
#\-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-/#
#  ---------------------------------------------------------

# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']

# Collect data from files
simulated_colours_fname = '{}-{}-{}-{}_minus_SDSS_corrections.csv'.format(telescope, instrument, source, filt)
simulated_colours_fname = os.path.join("Standard_colour_corrections", simulated_colours_fname)
print("Loading table {}".format(simulated_colours_fname))
simulated_colours_table = pd.read_csv(simulated_colours_fname)

smith_table_fname = "tab08.dat.txt"
smith_table = pd.read_fwf(smith_table_fname)

# Which one are we using?
cam_filters = super_filters if filt == 'super' else sdss_filters

# Construct the labels for the colour corrections. 
# These are also used as keys for what corrections to calculate and plot.
y_combinations = []
for sdss_filter, super_filter in zip(sdss_filters, cam_filters):
    y_combinations.append("{}-{}".format(super_filter, sdss_filter))

fig, axes = plt.subplots(
    len(y_combinations), len(x_combinations),
    figsize=figsize,
    sharex='col', sharey='row'
)
if axes.ndim < 2:
    axes = np.array([axes]).T

filter_labels = {
    "u_s": r"$u'_{sup}$",
    "g_s": r"$g'_{sup}$",
    "r_s": r"$r'_{sup}$",
    "i_s": r"$i'_{sup}$",
    "z_s": r"$z'_{sup}$",
    "u": r"$u'$",
    "g": r"$g'$",
    "r": r"$r'$",
    "i": r"$i'$",
    "z": r"$z'$",
    'u_s-u': "$u'_{sup} - u'$",
    'g_s-g': "$g'_{sup} - g'$",
    'r_s-r': "$r'_{sup} - r'$",
    'i_s-i': "$i'_{sup} - i'$",
    'z_s-z': "$z'_{sup} - z'$",
    'u-u': "$u'_{reg} - u'$",
    'g-g': "$g'_{reg} - g'$",
    'r-r': "$r'_{reg} - r'$",
    'i-i': "$i'_{reg} - i'$",
    'z-z': "$z'_{reg} - z'$",
    'u-g': "$u'-g'$",
    'g-r': "$g'-r'$",
    'r-i': "$r'-i'$",
    'i-z': "$i'-z'$",
}

# Set all the x labels
for j, x_col in enumerate(x_combinations):
    label = filter_labels[x_col]
    axes[-1][j].set_xlabel(label)

if tick_right:
    # Move ticks to the right side of the axes
    for row in axes:
        for cell in row:
            cell.yaxis.tick_right()

# Each colour track will have an associated GP, used to interpolate values from it.
# Storage for the Gaussian processes in a 2D list
gaussian_processes = [[] for _ in y_combinations]

# GP optimisation helper functions
def neg_ln_like(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p, gp, y):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)


# Plot the simulated mags
for i, y_col in enumerate(y_combinations):
    label = filter_labels[y_col]
    axes[i][0].set_ylabel(label)
    y_colours = simulated_colours_table[y_col]

    for j, x_col in enumerate(x_combinations):
        x_colours = simulated_colours_table[x_col]

        if source == 'pickles':
            axes[i][j].errorbar(
                x_colours, y_colours,
                yerr=[err for _ in y_colours],
                marker='.', linestyle='none', color='black'
            )

            # GP fit for Pickles
            x = x_colours
            y = y_colours
            yerr = np.var(y) ** 0.5
            yerr = yerr * np.ones_like(y)

            y_range = y.max() - y.min()
            x_range = x.max() - x.min()

            # Kernel black magic fuckery
            kernel = g.kernels.EmptyKernel()
            # kernel += g.kernels.LinearKernel(0.1, 1)
            kernel += g.kernels.Matern32Kernel(2.0)
            kernel += g.kernels.PolynomialKernel(log_sigma2=1.0, order=5)
            gp = g.GP(kernel)
            gp.compute(x, yerr)

            result = minimize(
                neg_ln_like,
                gp.get_parameter_vector(),
                args=(gp, y,),
                jac=grad_neg_ln_like
            )
            gp.set_parameter_vector(result.x)

            x_pred = np.linspace(x.min(), x.max(), 500)
            y_pred, y_pred_var = gp.predict(y, x_pred, return_var=True)

            axes[i][j].fill_between(
                x_pred,
                y_pred - np.sqrt(y_pred_var),
                y_pred + np.sqrt(y_pred_var),
                color='k', alpha=0.2
            )
            gaussian_processes[i].append(gp)

        if source == 'koester':
            axes[i][j] = simulated_colours_table.plot(
                ax=axes[i][j],
                kind='scatter',
                x=x_col, y=y_col,
                c='logg',
                colormap='viridis',
                colorbar=False
            )

# Interpolate the gaussian processes.
if not std_name.lower() == 'none' and source == 'pickles':
    std_data = smith_table.loc[smith_table['StarName']==std_name]

    for i, y_col in enumerate(y_combinations):
        y = simulated_colours_table[y_col]

        y_values = np.zeros_like(std_colours, dtype=np.float32)
        y_errs = np.zeros_like(std_colours, dtype=np.float32)
        print("For the correction {}".format(y_col))
        for j, x_col in enumerate(std_colours):
            x_colour = std_data[x_col].values[0]

            axes[i][j].axvline(x_colour, color='red')

            gp = gaussian_processes[i][j]
            y_pred, y_pred_var = gp.predict(y, x_colour, return_var=True)

            y_values[j] = y_pred[0]
            y_errs[j] = y_pred_var[0]
            print("  - From colour: {}, value {:.3f} Yields {:.3f}+/-{:.3f}".format(x_col, x_colour, y_pred[0], np.sqrt(y_pred_var)[0]))

            # Put blue lines where the GP thinks the colour corrections are
            axes[i][j].axhline(y_pred[-1], color='blue', alpha=0.7)
            axes[i][j].axhline(y_pred[-1] - np.sqrt(y_pred_var[-1]), color='blue', linestyle='--', alpha=0.6)
            axes[i][j].axhline(y_pred[-1] + np.sqrt(y_pred_var[-1]), color='blue', linestyle='--', alpha=0.6)

        ymean = y_values / y_errs
        ymean = np.sum(ymean) / np.sum(1.0/y_errs)
        ymean_err = np.sqrt(1.0/np.sum(1.0/y_errs))
        print("Overall, this mean is {:.3f}+/-{:.3f}".format(ymean, ymean_err))
        print("\n")

    if not publishable:
        fig.suptitle("{}/{}/{} - SDSS magnitudes\nStandard star {} is plotted.".format(
            telescope.upper(), instrument.upper(), filt, std_name)
        )
elif not publishable:
    fig.suptitle("{}/{} Colour corrections with SDSS".format(telescope.upper(), instrument.upper()))

if not publishable:
    fig.text(0.5, 0.04, 'SDSS Colours', ha='center')
    fig.text(0.04, 0.5, '{}/{}/{} - SDSS mags'.format(telescope,instrument,filt), va='center', rotation='vertical')


plt.tight_layout()
plt.subplots_adjust(hspace=0., wspace=0.)

output_figname = '{}-{}-{}_{}_{}_minus_SDSS_corrections.png'.format(
    telescope, instrument, source, std_name.replace(' ', ''), filt
)

plt.savefig(os.path.join("Standard_colour_corrections", output_figname))
plt.show()
