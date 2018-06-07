from astropy.io import fits
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import eunomia
import healpy as hp
import pandas as pd
import os
from chainconsumer import ChainConsumer

# Set important directories and create the figure directory if necessary.
data_dir = './test_data/'
fig_dir = './test_figs/'
out_dir = './test_out/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load in the convergence map.
hdu = fits.open(data_dir + 'kappa.fits')
tbdata = hdu[1].data
kappa = tbdata['signal'].reshape(-1)

# Determine nside and lmax.
npix = len(kappa)
nside = hp.npix2nside(npix)
lmax = 2 * nside

# Plot the convergence map.
hp.mollview(kappa, title='Flask Sim Convergence Map $n_{side}=%d$, $\ell_{max}=%d$' % (nside, lmax), unit='$\kappa$')
plt.savefig(fig_dir + 'kappa_map', dpi=300)

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'Cl.dat', delim_whitespace=True))[:, 1]

# Compute the full covariance matrix for the map from Cl's.
cov, ang_sep = eunomia.sim_tools.covariance.full_neighbor_cov_from_cl(cl, nside)

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

ms = eunomia.MapSampler(kappa, cov)
chain, logp = ms.sample(1000, 10, 0, 1.0)

plt.plot(range(len(logp)),logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

c = ChainConsumer()
c.add_chain(chain[:,:5])
c.plotter.plot(figsize="column")
plt.savefig(fig_dir + 'corner', dpi=300)