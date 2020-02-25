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
ucam_filt = 'regular'


# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']

ucam_filters = super_filters if ucam_filt == 'super' else sdss_filters
hcam_filters = super_filters if hcam_filt == 'super' else sdss_filters


# Lets store it all in a pandas dataframe.
INFO = ['SpecType', 'PicklesName']
SDSS_COLOURS = ["{}-{}".format(hcam_filters[i], hcam_filters[i+1]) for i in range(4)]
CORRECTIONS = ["{}-{}".format(a,b) for a,b in zip(hcam_filters, ucam_filters)]

COLNAMES = INFO + SDSS_COLOURS + CORRECTIONS
table = pd.DataFrame(columns=COLNAMES)



pickles_path = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk')

pickles_ms = (
    ('pickles_uk_1',    'O5V',     39810.7),
    ('pickles_uk_2',    'O9V',     35481.4),
    ('pickles_uk_3',    'B0V',     28183.8),
    ('pickles_uk_4',    'B1V',     22387.2),
    ('pickles_uk_5',    'B3V',     19054.6),
    ('pickles_uk_6',    'B5-7V',   14125.4),
    ('pickles_uk_7',    'B8V',     11749.0),
    ('pickles_uk_9',    'A0V',     9549.93),
    ('pickles_uk_10',   'A2V',     8912.51),
    ('pickles_uk_11',   'A3V',     8790.23),
    ('pickles_uk_12',   'A5V',     8491.80),
    ('pickles_uk_14',   'F0V',     7211.08),
    ('pickles_uk_15',   'F2V',     6776.42),
    ('pickles_uk_16',   'F5V',     6531.31),
    ('pickles_uk_20',   'F8V',     6039.48),
    ('pickles_uk_23',   'G0V',     5807.64),
    ('pickles_uk_26',   'G2V',     5636.38),
    ('pickles_uk_27',   'G5V',     5584.70),
    ('pickles_uk_30',   'G8V',     5333.35),
    ('pickles_uk_31',   'K0V',     5188.00),
    ('pickles_uk_33',   'K2V',     4886.52),
    ('pickles_uk_36',   'K5V',     4187.94),
    ('pickles_uk_37',   'K7V',     3999.45),
    ('pickles_uk_38',   'M0V',     3801.89),
    ('pickles_uk_40',   'M2V',     3548.13),
    ('pickles_uk_43',   'M4V',     3111.72),
    ('pickles_uk_44',   'M5V',     2951.21)
)


# Get the SDSS colours
for name, spec, teff in pickles_ms:
    row = {
        'Teff': teff,
        'SpecType': spec,
        'PicklesName': name
    }

    # Unset the *CAM thruput stuff
    S.setref(comptable=None, graphtable=None)  # leave the telescope area as it was
    S.setref(area=None)  # reset the telescope area as well

    # Synthetic spectrum
    sp = S.FileSpectrum(os.path.join(pickles_path, name+'.fits'))

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
            c='Teff',
            colormap='hot',
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
table_oname = "hcam_{}_{}-ucam_{}_{}-pickles.csv".format(hcam_tel, hcam_filt, ucam_tel, ucam_filt)
table.to_csv(os.path.join("hcam-ucam_corrections", table_oname), index=False)
print("\n\nWrote table to:\n{}".format(table_oname))

output_figname = "hcam_{}_{}-ucam_{}_{}-pickles.png".format(hcam_tel, hcam_filt, ucam_tel, ucam_filt)
plt.savefig(os.path.join("hcam-ucam_corrections", output_figname))
plt.show()