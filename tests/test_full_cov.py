from astropy.io import fits
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import eunomia
import healpy as hp
import pandas as pd
import os

# Set important directories and create the figure directory if necessary.
data_dir = './test_data/'
fig_dir = './test_figs/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Load in the convergence map.
hdu = fits.open(data_dir + 'kappa.fits')
tbdata = hdu[1].data
kappa = tbdata['signal'].reshape(-1)

# Determine nside and lmax.
nside = hp.npix2nside(len(kappa))
lmax = 2 * nside

# Plot the convergence map.
hp.mollview(kappa, title='Flask Sim Convergence Map $n_{side}=%d$, $\ell_{max}=%d$' % (nside, lmax), unit='$\kappa$')
plt.savefig(fig_dir + 'kappa_map', dpi=300)

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'Cl.dat', delim_whitespace=True))[:, 1]

# Compute the full covariance matrix for the map from Cl's.
cov = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside)

# Plot the covariance matrix.
plt.matshow(cov, norm=matplotlib.colors.LogNorm())
cbar = plt.colorbar()
plt.suptitle('Full $\kappa$ Covariance')
plt.savefig(fig_dir + 'kappa_full_cov', dpi=300)

# Show a zoomed in plot of the matrix to show structure.
plt.matshow(cov[:100, :100], norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.suptitle('Full $\kappa$ Covariance (Zoomed)')
plt.savefig(fig_dir + 'kappa_full_cov_zoomed', dpi=300)
