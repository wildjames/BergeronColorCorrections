import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import george as g

# Variables
std_name = 'SA 114 548'
#std_name = 'BD-21 0910'
#std_name = 'BD+17 4708'
#std_name = 'BD+33 2642'

# std_name = 'None'

telescope = 'ntt'
instrument = 'ucam'
filt = 'super'

source = 'pickles'



# Collect data from files
simulated_colours_fname = '{}-{}-{}-{}_minus_SDSS_corrections.csv'.format(telescope, instrument, source, filt)
simulated_colours_table = pd.read_csv(os.path.join("Standard_colour_corrections", simulated_colours_fname))

smith_table_fname = "tab08.dat.txt"
smith_table = pd.read_fwf(smith_table_fname)


# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']
# Which one are we using?
cam_filters = super_filters if filt == 'super' else sdss_filters


y_combinations = []
for sdss_filter, super_filter in zip(sdss_filters, cam_filters):
    y_combinations.append("{}-{}".format(super_filter, sdss_filter))
x_combinations = ['u-g', 'g-r', 'r-i', 'i-z']


fig, axes = plt.subplots(
    len(y_combinations), len(x_combinations)+1,
    figsize=(16,9),
    sharex='col', sharey='row'
)

# Set all the x labels
for j, x_col in enumerate(x_combinations):
    axes[-1][j].set_xlabel(x_col)

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

# Error to add to the y-axis. The Smith values have errors <0.005 mags,
# so subtracted their max error is 0.010. This will do as a rule of thumb for the Pickles data
err = 0.010

# Plot the simulated mags
for i, y_col in enumerate(y_combinations):
    axes[i][0].set_ylabel(y_col)
    y_colours = simulated_colours_table[y_col]

    for j, x_col in enumerate(x_combinations):
        x_colours = simulated_colours_table[x_col]

        if source == 'pickles':
            axes[i][j].errorbar(
                x_colours, y_colours,
                yerr=[err for _ in y_colours],
                marker='.', linestyle='none', color='black'
            )

        if source == 'koester':
            axes[i][j] = simulated_colours_table.plot(
                ax=axes[i][j],
                kind='errorbar',
                x=x_col, y=y_col, yerr=[err for _ in y_col],
                c='logg',
                colormap='viridis',
                colorbar=False
            )

        # GP fit
        x = x_colours
        y = y_colours
        yerr = np.var(y) ** 0.5
        yerr = yerr * np.ones_like(y)

        y_range = y.max() - y.min()
        x_range = x.max() - x.min()

        kernel = g.kernels.EmptyKernel()
        kernel += g.kernels.LinearKernel(0.1, 1)
        kernel += g.kernels.Matern32Kernel(1.0)
        kernel += g.kernels.PolynomialKernel(log_sigma2=1.0, order=2)
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
            color='k', alpha=0.3
        )

        gaussian_processes[i].append(gp)



if not std_name.lower() == 'none':
    std_data = smith_table.loc[smith_table['StarName']==std_name]

    std_colours = ["u'-g'", "g'-r'", "r'-i'", "i'-z'"]
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

            axes[i][j].axhline(y_pred[-1], color='blue', alpha=0.4)
            axes[i][j].axhline(y_pred[-1] - np.sqrt(y_pred_var[-1]), color='blue', linestyle='--', alpha=0.3)
            axes[i][j].axhline(y_pred[-1] + np.sqrt(y_pred_var[-1]), color='blue', linestyle='--', alpha=0.3)

            axes[i][-1].errorbar([0.0], [y_pred[-1]], yerr=[np.sqrt(y_pred_var[-1])], fmt='b.')

        ymean = y_values / y_errs
        ymean = np.sum(ymean) / np.sum(1.0/y_errs)
        ymean_err = np.sqrt(1.0/np.sum(1.0/y_errs))
        print("Overall, this mean is {:.3f}+/-{:.3f}".format(ymean, ymean_err))
        print("\n")

        axes[i][-1].errorbar([0.0], [ymean], yerr=[ymean_err], fmt='r.')
        axes[i][-1].annotate("Mean:\n{:.3f}+/-{:.3f}".format(ymean, ymean_err),
                 (0.0,ymean),                    # this is the point to label
                 textcoords="offset points",     # how to position the text
                 xytext=(50,0),                  # distance from text to points (x,y)
                 ha='center'                     # horizontal alignment can be left, right or center
        )

    fig.suptitle("{}/{}/{} - SDSS magnitudes\nStandard star {} is plotted.".format(telescope.upper(), instrument.upper(), filt, std_name))

else:
    fig.suptitle("{}/{} Colour corrections with SDSS".format(telescope.upper(), instrument.upper()))

fig.text(0.5, 0.04, 'SDSS Colours', ha='center')
fig.text(0.04, 0.5, '{}/{}/{} - SDSS mags'.format(telescope,instrument,filt), va='center', rotation='vertical')

plt.subplots_adjust(
    hspace=0.0, wspace=0.0,
    top=0.9, bottom=0.1, left=0.1, right=0.9
)


output_figname = '{}-{}-{}_{}_{}_minus_SDSS_corrections.png'.format(telescope, instrument, source, std_name.replace(' ', ''), filt)
plt.savefig(os.path.join("Standard_colour_corrections", output_figname))
plt.show()
