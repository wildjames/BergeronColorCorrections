'''Construct a table of traditional, u-g etc. colours, and corrections to the
UCAM 'frame'. This table will be interpolated (fitted with colour to get a Teff,
then use that Teff to retrieve corrections) for a specific star to get its
expected observed magnitude through the *CAMs.
'''

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysynphot as S
from matplotlib import pyplot as plt
from ucam_thruput import getref, setup

from generate_mags import get_teff_logg

# Lightpath data
instrument = 'ucam'
telescope = 'ntt'
filt = 'regular'



# SDSS filters
sdss_filters = ['u', 'g', 'r', 'i', 'z']
# Super SDSS filters
super_filters = ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']

cam_filters = super_filters if filt == 'super' else sdss_filters

# Lets store it all in a pandas dataframe.
INFO = ['Teff', 'logg']
SDSS_COLOURS = ['u-g', 'g-r', 'r-i', 'i-z']
CORRECTIONS = ["{}-{}".format(a,b) for a,b in zip(cam_filters, sdss_filters)]

COLNAMES = INFO + SDSS_COLOURS + CORRECTIONS
table = pd.DataFrame(columns=COLNAMES)


# I want to do each file in the koester catalogue
for filename in Path('koester2/').rglob("*.txt"):
    # Unset the *CAM thruput stuff
    S.setref(comptable=None, graphtable=None)  # leave the telescope area as it was
    S.setref(area=None)  # reset the telescope area as well

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

    # Get all the SDSS magnitudes
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

    # Magnitudes
    for f in cam_filters:
        bp = S.ObsBandpass("{},{},{}".format(telescope,instrument,f))
        obs = S.Observation(sp, bp, force='taper')
        mag = obs.effstim("abmag")
        simulated_mags[f] = mag

    # Colours
    for colour in CORRECTIONS:
        f1, f2 = colour.split('-')
        correction_mag = simulated_mags[f1] - simulated_mags[f2]

        row[colour] = correction_mag

    # Update the table
    table = table.append(row, ignore_index=True, sort=True)

# This is where I'll save the table.
table_oname = "{}-{}-koester-{}_minus_SDSS_corrections.csv".format(telescope, instrument, filt)
table.to_csv(os.path.join("Standard_colour_corrections", table_oname), index=False)
print("\n\nWrote table to:\n{}".format(table_oname))
