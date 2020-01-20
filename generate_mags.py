import os
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import multiprocessing
from functools import partial

import pysynphot as S
from ucam_thruput import getref, _make_instrument_reference_table


def get_teff_logg(fname):
    fname = os.path.split(fname)[1]
    teff = int(fname[2:7])
    logg = int(fname[8:11])

    return teff, logg

def get_fname(teff, logg):
    teff = str(teff)
    logg = str(logg)
    return "koester2/da{:>05s}_{:>03s}.dk.dat.txt".format(teff, logg)

##### IS THIS EQUATION THE SAME FOR SUPER SDSS AND REGULAR? #####
def sdss_mag2flux(mag):
    '''Takes an SDSS magnitude, returns the corresponding flux in [mJy]'''
    alpha = 3631e3

    flux = 10**(-mag/2.5)
    flux*= alpha

    return flux

def sdss_flux2mag(flx):
    '''Takes an flux in mJy and converts it to an SDSS magnitude'''
    alpha = 3631e3

    m = -2.5 * np.log10(flx)
    m += 2.5 * np.log10(alpha)

    return m

def get_col_correction(telescope, instrument, filter, teff, logg, plot=False):
    S.setref(**getref(telescope))
    bp = S.ObsBandpass('{},{},{}'.format(telescope, instrument, filter))
    bp_HCAM_GTC = S.ObsBandpass('hcam,gtc,{}'.format(filter.split('_')[0] + '_s'))

    table_fname = get_fname(teff, logg)
    teff, logg = get_teff_logg(table_fname)
    table = pd.read_csv(
        table_fname,
        delim_whitespace=True,
        comment='#',
        names=['WAVELENGTH (ANGSTROM)', 'FLUX (ERG/CM2/S/A)']
    )

    # drop duplicate wavelengths. WTF are they here Koester?
    table.drop_duplicates(subset='WAVELENGTH (ANGSTROM)', inplace=True)

    # create pysynphot spectrum
    sp = S.ArraySpectrum(table['WAVELENGTH (ANGSTROM)'], table['FLUX (ERG/CM2/S/A)'])
    obs = S.Observation(sp, bp, force='taper')
    obs_HCAM_GTC = S.Observation(sp, bp_HCAM_GTC, force='taper')
    tot_mag = obs.effstim('abmag')

    col_term = obs_HCAM_GTC.effstim('abmag') - tot_mag

    if plot:
        # Plotting stuff
        limits = (bp.wave.min(), bp.wave.max())
        cut_table = table[(table['WAVELENGTH (ANGSTROM)'] < limits[1]) & (table['WAVELENGTH (ANGSTROM)'] > limits[0])]

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax2.plot(bp.wave, bp.throughput, color='red', linestyle='--', label='{}, {} Throughput'.format(instrument, telescope))
        ax2.plot(bp_HCAM_GTC.wave, bp_HCAM_GTC.throughput, color='black', linestyle='--', label='HCAM, GTC Throughput')
        ax.plot(obs.wave, obs.flux, color='red', label='{}, {} stimulation'.format(instrument, telescope))
        ax.plot(obs_HCAM_GTC.wave, obs_HCAM_GTC.flux, color='black', label='HCAM, GTC stimulation')
        ax.step(cut_table['WAVELENGTH (ANGSTROM)'], cut_table['FLUX (ERG/CM2/S/A)'], color='lightgrey', label='Raw Spectrum')

        ax.set_title("Teff: {} || log(g): {} || detected magnitude: {:.3f} || Color {:.3f}".format(teff, logg, tot_mag, col_term))
        ax.set_xlabel("Wavelength, Angstroms")
        ax.set_ylabel("Flux, erg/cm2/s/A")
        ax.set_xlim(limits)
        fig.legend()
        plt.tight_layout()

        plt.show()

    return col_term

def parse_row(fname, filters, telescope, instrument):
    teff, logg = get_teff_logg(fname)
    row = {
        'Teff': teff,
        'logg': logg/100.
    }

    for reg in filters:
        col_term = get_col_correction(telescope, instrument, reg, teff, logg, False)
        row[reg] = col_term
    return row

def construct_table(telescope, instrument, filters):
    '''Computes the ugriz and u'g'r'i'z' magnitudes, and the difference between
    each filter and its hipercam, GTC, super SDSS counterpart, for each file in koester2.'''

    # Create an empty dataframe
    cols = ['Teff', 'logg'] + filters

    mags = pd.DataFrame(
        columns=cols
    )

    # I want to do each file in the koester catalogue
    table_fnames = []
    for filename in Path('koester2/').rglob("*.txt"):
        table_fnames.append(filename)

    # Helper to pass multiple arguments to multiprocessing, when some are fixed
    prod = partial(
        parse_row,
        filters=filters,
        telescope=telescope, instrument=instrument,
    )

    # Crank through all the files in parallel
    pool = multiprocessing.Pool(8)
    rows = pool.map(prod, table_fnames)
    pool.close()
    pool.join()

    # Stitch that back together, and save it to file
    mags = pd.DataFrame(rows)
    mags = mags.sort_values(by=['Teff', 'logg'])

    for filt in filters:
        ax = mags.plot(
            kind='scatter',
            x='Teff', y=filt,
            c='logg',
            colormap='viridis',
            title='Koester Color Corrections {} on {}'.format(telescope, instrument),
        )
        plt.tight_layout()
        plt.savefig("color_corrections_HCAM-GTC_minus_{}-on-{}_{}.pdf".format(instrument, telescope, filt))
        plt.close()

    oname = "calculated_mags_{}_{}.csv".format(telescope, instrument)
    print("Done all files. Saving to {}...".format(oname))
    mags.to_csv(oname, index=False, index_label=False)
    return mags


if __name__ == "__main__":
    telescope = 'ntt'
    inst = 'ucam'
    filters = ['u', 'g', 'r', 'i', 'z', 'u_s', 'g_s', 'r_s', 'i_s', 'z_s']
    construct_table(telescope, inst, filters)

    telescope = 'tnt'
    inst = 'uspec'
    filters = ['u', 'g', 'r', 'i', 'z']
    construct_table(telescope, inst, filters)


