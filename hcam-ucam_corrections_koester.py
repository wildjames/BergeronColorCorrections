import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysynphot as S
from matplotlib import pyplot as plt
from ucam_thruput import getref, setup

from generate_mags import get_teff_logg


# Lightpath data
hcam_tel = 'gtc'
ucam_tel = 'ntt'
hcam_filt = 'super'
ucam_filt = 'super'


# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']

ucam_filters = super_filters if ucam_filt == 'super' else sdss_filters
hcam_filters = super_filters if hcam_filt == 'super' else sdss_filters


# Lets store it all in a pandas dataframe.
INFO = ['Teff', 'logg']
SDSS_COLOURS = ["{}-{}".format(hcam_filters[i], hcam_filters[i+1]) for i in range(4)]
CORRECTIONS = ["{}-{}".format(a,b) for a,b in zip(hcam_filters, ucam_filters)]

COLNAMES = INFO + SDSS_COLOURS + CORRECTIONS
table = pd.DataFrame(columns=COLNAMES)

# I want to do each file in the koester catalogue
for filename in Path('koester2/').rglob("*.txt"):
    teff, logg = get_teff_logg(filename)

    row = {
        'Teff': teff,
        'logg': float(logg)/100.0,
    }

    teff, logg = get_teff_logg(filename)
    koester_spectrum = pd.read_csv(
        filename,
        delim_whitespace=True,
        comment='#',
        names=['WAVELENGTH (ANGSTROM)', 'FLUX (ERG/CM2/S/A)']
    )

    # drop duplicate wavelengths. WTF are they here Koester?
    koester_spectrum.drop_duplicates(subset='WAVELENGTH (ANGSTROM)', inplace=True)

    # create pysynphot spectrum
    sp = S.ArraySpectrum(
        koester_spectrum['WAVELENGTH (ANGSTROM)'],
        koester_spectrum['FLUX (ERG/CM2/S/A)'],
        fluxunits='flam',
        waveunits='Angstroms'
    )

    # Set the lightpath for the hcam observations
    S.setref(**getref(hcam_tel))

    # Get all the hcam magnitudes
    simulated_mags = {}
    for f in hcam_filters:
        bp = S.ObsBandpass("{},{},{}".format('hcam',hcam_tel, f))
        obs = S.Observation(sp, bp, force='taper')
        mag = obs.effstim("abmag")
        simulated_mags['hcam_{}'.format(f)] = mag

    # Get the actual colours
    for colour in SDSS_COLOURS:
        f1, f2 = colour.split("-")
        colour_mag = simulated_mags["hcam_{}".format(f1)] - simulated_mags["hcam_{}".format(f2)]

        row[colour] = colour_mag

    # Magnitudes
    for f in ucam_filters:
        bp = S.ObsBandpass("{},{},{}".format(ucam_tel,'ucam',f))
        obs = S.Observation(sp, bp, force='taper')
        mag = obs.effstim("abmag")
        simulated_mags['ucam_{}'.format(f)] = mag

    # Colours
    for colour in CORRECTIONS:
        f1, f2 = colour.split('-')
        correction_mag = simulated_mags["hcam_{}".format(f1)] - simulated_mags["ucam_{}".format(f2)]

        row[colour] = correction_mag

    # Update the table
    table = table.append(row, ignore_index=True, sort=True)



y_combinations = []
for hcam_filter, ucam_filter in zip(hcam_filters, ucam_filters):
    y_combinations.append("{}-{}".format(hcam_filter, ucam_filter))
x_combinations = SDSS_COLOURS

fig, axes = plt.subplots(
    len(y_combinations), len(x_combinations),
    figsize=(16,12),
    sharex='col', sharey='row'
)

# Plot the simulated mags
for i, y_col in enumerate(y_combinations):
    axes[i][0].set_ylabel(y_col)

    for j, x_col in enumerate(x_combinations):
        x_colours = table[x_col]
        y_colours = table[y_col]

        axes[i][j] = table.plot(
            ax=axes[i][j],
            kind='scatter',
            x=x_col, y=y_col,
            c='logg',
            colormap='viridis',
            colorbar=False
        )

fig.suptitle("HCAM/{}/{} - UCAM/{}/{} corrections".format(hcam_tel, hcam_filt, ucam_tel, ucam_filt))

fig.text(0.5, 0.04, 'GTC Colours', ha='center')
fig.text(0.04, 0.5, 'HCAM - UCAM Colours', va='center', rotation='vertical')

plt.subplots_adjust(
    hspace=0.0, wspace=0.0,
    top=0.9, bottom=0.1, left=0.1, right=0.9
)


# This is where I'll save the table.
table_oname = "hcam_{}_{}-ucam_{}_{}-koester.csv".format(hcam_tel, hcam_filt, ucam_tel, ucam_filt)
table.to_csv(os.path.join("hcam-ucam_corrections", table_oname), index=False)
print("\n\nWrote table to:\n{}".format(table_oname))

output_figname = "hcam_{}_{}-ucam_{}_{}-koester.png".format(hcam_tel, hcam_filt, ucam_tel, ucam_filt)
plt.savefig(os.path.join("hcam-ucam_corrections", output_figname))
plt.show()