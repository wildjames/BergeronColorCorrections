import os

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysynphot as S
from ucam_thruput import getref

from generate_mags import get_fname, get_teff_logg

observations = ['SDSS', 'UCAM']
sources = ['pickles', 'SDSS', 'koester']


# The numbers in Pickles:
# http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas

# # HD 121968 - B2ii star http://simbad.u-strasbg.fr/simbad/sim-id?Ident=HD+121968&submit=submit+id
# N = 106
# source = 'pickles'

# # SA 99-438- B3/5iii star http://simbad.u-strasbg.fr/simbad/sim-id?Ident=SA+99-438&submit=submit+id
# N = 5
# source = 'pickles'

# # G162-66 - DA1.9 White Dwarf http://simbad.u-strasbg.fr/simbad/sim-id?Ident=G162-66&submit=submit+id
# # The SDSS observed spectrum. Doesn't cover the u.
# fits_image_filename = "spec-3832-55289-0169.fits"
# source = 'SDSS'
# # Model WD spectrum
# teff = 26000
# logg = '800'
# source = 'koester'

# # Feige22
# teff = 20000
# logg = "850"
# source = 'koester'

# # Ross 288 - K0 with D confidence in SIMBAD.
# source = 'pickles'
# N = 57  # K0iv
# N = 78  # K0iii
# N = 111 # K0-1ii
# N = 31  # K0v

# # SA 98-185 - A2 with confidence E
# source = 'pickles'
# N = 10

# # SA 105-815 - sdF0 star with D confidence. Using a main sequence F0V spectrum, hopefully it's close enough.
# source = 'pickles'
# N = 14

# SA 114-548
source = 'pickles'
N = 33 # K2V
N = 128 # K2I


# The filters we care about
filters = ["u", 'g', 'r', 'i', 'z']
filters = ["u_s", 'g_s', 'r_s', 'i_s', 'z_s']

# how is it observed?
observation = 'UCAM'




######################################################################

if source == 'pickles':
    # Get the pickles spectrum from pysynphot.
    filename = os.path.join(
        os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk', 'pickles_uk_{}.fits'.format(N))
    sp = S.FileSpectrum(filename)

if source == 'SDSS':
    hdul = fits.open(fits_image_filename)
    data = hdul[1].data

    # a right pain in the arse to extract.
    loglam = np.array([d[1] for d in data])
    wav = 10.0**loglam
    flx = np.array([d[0] for d in data])

    sp = S.ArraySpectrum(wave=wav, flux=flx, waveunits='Angstroms', fluxunits='flam')

if source == 'koester':
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
    sp = S.ArraySpectrum(
        table['WAVELENGTH (ANGSTROM)'],
        table['FLUX (ERG/CM2/S/A)'],
        fluxunits='flam',
        waveunits='Angstroms'
    )




# Storage dict
mags = {}

if observation == 'UCAM':
    # This is the bandpass in the relevant lightpath
    telescope, instrument = "ntt", "ucam"
    # Setup the *CAM
    S.setref(**getref(telescope))

    for filt in filters:
        # Get the lightpath
        bp = S.ObsBandpass('{},{},{}'.format(telescope, instrument, filt))

        # Run the spectrum through the bandpass. Should include atmosphere?
        obs = S.Observation(sp, bp, force="taper")

        # Convert to magnitude
        ADU_flux = obs.countrate()
        mag = obs.effstim('abmag')

        mags[filt] = mag

        print("Band: {}".format(filt))
        print("Countrate: {:.3f}".format(ADU_flux))
        print("abmag: {:.3f}\n".format(mag))

elif observation == 'SDSS':
    for filt in filters:
        # Get the lightpath
        bp = S.ObsBandpass("sdss,{}".format(filt))

        # Run the spectrum through the bandpass. Should include atmosphere?
        obs = S.Observation(sp, bp, force="taper")

        # Convert to magnitude
        ADU_flux = obs.countrate()
        mag = obs.effstim('abmag')

        mags[filt] = mag

        print("Band: {}".format(filt))
        print("Countrate: {:.3f}".format(ADU_flux))
        print("abmag: {:.3f}\n".format(mag))


print(filters)
for filt in filters:
    print("{:.3f}".format(mags[filt]))
