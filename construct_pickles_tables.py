'''Construct a table of traditional, u-g etc. colours, and corrections to the
UCAM 'frame'. This table will be interpolated (fitted with colour to get a Teff,
then use that Teff to retrieve corrections) for a specific star to get its
expected observed magnitude through the *CAMs.
'''

import os

import numpy as np
import pandas as pd
import pysynphot as S
from matplotlib import pyplot as plt
from ucam_thruput import getref, setup


pickles_path = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk')
pickles_ms = (
    ('pickles_uk_1',    'O5V'   ),
    ('pickles_uk_2',    'O9V'   ),
    ('pickles_uk_3',    'B0V'   ),
    ('pickles_uk_4',    'B1V'   ),
    ('pickles_uk_5',    'B3V'   ),
    ('pickles_uk_6',    'B5-7V' ),
    ('pickles_uk_7',    'B8V'   ),
    ('pickles_uk_9',    'A0V'   ),
    ('pickles_uk_10',   'A2V'   ),
    ('pickles_uk_11',   'A3V'   ),
    ('pickles_uk_12',   'A5V'   ),
    ('pickles_uk_14',   'F0V'   ),
    ('pickles_uk_15',   'F2V'   ),
    ('pickles_uk_16',   'F5V'   ),
    ('pickles_uk_20',   'F8V'   ),
    ('pickles_uk_23',   'G0V'   ),
    ('pickles_uk_26',   'G2V'   ),
    ('pickles_uk_27',   'G5V'   ),
    ('pickles_uk_30',   'G8V'   ),
    ('pickles_uk_31',   'K0V'   ),
    ('pickles_uk_33',   'K2V'   ),
    ('pickles_uk_36',   'K5V'   ),
    ('pickles_uk_37',   'K7V'   ),
    ('pickles_uk_38',   'M0V'   ),
    ('pickles_uk_40',   'M2V'   ),
    ('pickles_uk_43',   'M4V'   ),
    ('pickles_uk_44',   'M5V'   )
)


# Lightpath data
instrument = 'ucam'
telescope = 'ntt'
filt = 'super'


# Apply extinction to that spectrum
a_g = 0.561
a_g = 0.0

# Convert from Gaia Ag to EBV
a_v = a_g/0.789
ebv = a_v/3.1
ext = S.Extinction(ebv, 'gal3')

# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']

cam_filters = super_filters if filt == 'super' else sdss_filters

# Lets store it all in a pandas dataframe.
INFO = ['SpecType', 'PicklesName']
SDSS_COLOURS = ['u-g', 'g-r', 'r-i', 'i-z']
CORRECTIONS = ["{}-{}".format(a,b) for a,b in zip(cam_filters, sdss_filters)]

COLNAMES = INFO + SDSS_COLOURS + CORRECTIONS
table = pd.DataFrame(columns=COLNAMES)


# Get the SDSS colours
for name, spec in pickles_ms:
    row = {
        'SpecType': spec,
        'PicklesName': name
    }

    # Unset the *CAM thruput stuff
    S.setref(comptable=None, graphtable=None)  # leave the telescope area as it was
    S.setref(area=None)  # reset the telescope area as well

    # Synthetic spectrum
    sp = S.FileSpectrum(os.path.join(pickles_path, name+'.fits'))

    # Apply extinction to that spectrum
    ext = S.Extinction(ebv, 'gal3')
    sp = sp*ext

    # Get all the magnitudes
    simulated_mags = {}
    for f in sdss_filters:
        bp = S.ObsBandpass("{},{}".format('sdss', f))
        obs = S.Observation(sp, bp, force='taper')
        mag = obs.effstim("abmag")
        simulated_mags[f] = mag

    # Get the actual colours
    for colour in SDSS_COLOURS:
        f1, f2 = colour.split("-")
        colour_mag = simulated_mags[f1] - simulated_mags[f2]

        row[colour] = colour_mag

    # Get the colour corrections for the super filters
    S.setref(**getref(telescope))

    for f in cam_filters:
        bp = S.ObsBandpass("{},{},{}".format(telescope,instrument,f))
        obs = S.Observation(sp, bp, force='taper')
        mag = obs.effstim("abmag")
        simulated_mags[f] = mag

    for colour in CORRECTIONS:
        f1, f2 = colour.split('-')
        correction_mag = simulated_mags[f1] - simulated_mags[f2]

        row[colour] = correction_mag

    # Update the table
    table = table.append(row, ignore_index=True, sort=True)

# This is where I'll save the table.
table_oname = "{}-{}-pickles-{}_minus_SDSS_corrections.csv".format(telescope, instrument, filt)
table.to_csv(os.path.join("Standard_colour_corrections", table_oname), index=False)
print("\n\nWrote table to:\n{}".format(table_oname))


