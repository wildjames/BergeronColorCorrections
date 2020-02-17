import os

import matplotlib.pyplot as plt
import numpy as np
import pysynphot as S
from ucam_thruput import _make_instrument_reference_table, getref

# This is the bandpass in the relevant lightpath
telescope, instrument = "gtc", "hcam"
filter = "g_s"

# The number in Pickles - http://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas
N = 106
plax = 0.2921

S.setref(**getref(telescope))
bp = S.ObsBandpass('{},{},{}'.format(telescope, instrument, filter))


# Get the pickles spectrum from pysynphot.
filename = os.path.join(
    os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk', 'pickles_uk_{}.fits'.format(N))
orig_sp = S.FileSpectrum(filename)


# Move to the distance that star is at, assuming the above is at the star's surface.
dist = 1000.0/plax
dmod = 5.0 * np.log10(dist / 10.0)
sp = orig_sp.addmag(dmod)

# Run the spectrum through the bandpass
obs = S.Observation(sp, bp, force="taper")

# Convert to magnitude
ADU_flux = obs.countrate()
mag = -2.5 * np.log10(ADU_flux)
print("\nCountrate: {:.3f}".format(ADU_flux))
print("Instrumental mag: {:.3f}\n".format(mag))

