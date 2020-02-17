import os

import matplotlib.pyplot as plt
import numpy as np
import pysynphot as S
from ucam_thruput import _make_instrument_reference_table, getref

# This is the bandpass in the relevant lightpath
telescope, instrument = "ntt", "ucam"
filters = ["u", 'g', 'r', 'i', 'z']

# The number in Pickles - http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas

# HD 121968 - B2ii star http://simbad.u-strasbg.fr/simbad/sim-id?Ident=HD+121968&submit=submit+id
N = 106
plax = 0.2921


S.setref(**getref(telescope))

# Get the pickles spectrum from pysynphot.
filename = os.path.join(
    os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk', 'pickles_uk_{}.fits'.format(N))
orig_sp = S.FileSpectrum(filename)

# Move to the distance that star is at, assuming the above is at the star's surface.
dist = 1000.0/plax
dmod = 5.0 * np.log10(dist / 10.0)
sp = orig_sp.addmag(dmod)

# TODO: Do I need to integrate over the star's profile?

mags = {}
for filt in filters:
    # Get the lightpath
    bp = S.ObsBandpass('{},{},{}'.format(telescope, instrument, filt))
    # Run the spectrum through the bandpass
    obs = S.Observation(sp, bp, force="taper")

    # Convert to magnitude
    ADU_flux = obs.countrate()
    mag = obs.effstim('abmag')

    mags[filt] = mag

    print("Band: {}".format(filt))
    print("Countrate: {:.3f}".format(ADU_flux))
    print("abmag: {:.3f}\n".format(mag))

ug = mags['u'] - mags['g']
gr = mags['g'] - mags['r']
ri = mags['r'] - mags['i']
iz = mags['i'] - mags['z']

print("r: {:.3f}".format(mags['r']))
print("u - g: {:.3f}".format(ug))
print("g - r: {:.3f}".format(gr))
print("r - i: {:.3f}".format(ri))
print("i - z: {:.3f}".format(iz))
