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

# Determine nside and lmax.
nside = 16
lmax = 2 * nside

# Load in the convergence map.
kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))

k = kappas[:, 0]
kn = kappas_noise[:, 0]
g1_obs, g2_obs = eunomia.sim_tools.shear_conv_transformations.conv2shear(kn, lmax)

# Plot the convergence map.
hp.mollview(kn, title='Flask Sim Convergence Map $n_{side}=%d$, $\ell_{max}=%d$' % (nside, lmax), unit='$\kappa$')
plt.savefig(fig_dir + 'kappa_map', dpi=300)

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'Cl.dat', delim_whitespace=True))[:, 1]

# Compute the full covariance matrix for the map from Cl's.
shift = 0.053
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside)
theory_cov = np.log(1 + ln_theory_cov / (shift ** 2))

# # Plot the covariance matrix.
# plt.matshow(cov, norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar()
# plt.suptitle('Full $\kappa$ Covariance')
# plt.savefig(fig_dir + 'kappa_full_cov', dpi=300)
#
# # Show a zoomed in plot of the matrix to show structure.
# plt.matshow(cov[:100, :100], norm=matplotlib.colors.LogNorm())
# plt.colorbar()
# plt.suptitle('Full $\kappa$ Covariance (Zoomed)')
# plt.savefig(fig_dir + 'kappa_full_cov_zoomed', dpi=300)

k2g1, k2g2 = eunomia.sim_tools.shear_conv_transformations.compute_full_conv2shear_mats(nside)

ms = eunomia.MapSampler(g1_obs, g2_obs, k2g1, k2g2, shift, theory_cov)
chain, logp = ms.sample(1000, 10, 0, 1.0)

plt.plot(range(len(logp)), logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

c = ChainConsumer()
c.add_chain(chain[:, :5])
c.plotter.plot(figsize="column")
plt.savefig(fig_dir + 'corner', dpi=300)
