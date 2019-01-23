from astropy.io import fits
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import eunomia
import healpy as hp
import pandas as pd
import os
from chainconsumer import ChainConsumer
from tqdm import tqdm

# Set important directories and create the figure directory if necessary.
data_dir = './test_data/'
fig_dir = './test_figs/'
out_dir = './test_out/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Determine nside and lmax.
nside = 128
lmax = 2 * nside

# Load in the convergence map.
kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))
gammas_noise = np.load(data_dir + 'gammas_noise_{0}_{1}.npy'.format(nside, lmax))
# kappas_noise = kappas.copy()

# nside = 16
# lmax = 2 * nside

# kappas = hp.ud_grade(kappas.T, nside).T
# kappas_noise = kappas.copy()

k = kappas[:, 0]
kn = kappas_noise[:, 0]
# g1_obs, g2_obs = eunomia.sim_tools.shear_conv_transformations.conv2shear(k, lmax)
g1_obs = gammas_noise[0, :, 0]
g2_obs = gammas_noise[1, :, 0]

# Plot the convergence map.
hp.mollview(kn, title='Flask Sim Convergence Map $n_{side}=%d$, $\ell_{max}=%d$' % (nside, lmax), unit='$\kappa$')
plt.savefig(fig_dir + 'kappa_map', dpi=300)

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True))[:, 1]
pixwin = hp.pixwin(nside)
cl_pw = cl[:len(pixwin)] * (pixwin ** 2)
# cl = cl[:lmax + 1] * (pixwin[:lmax + 1] ** 2)

# kd = hp.ud_grade(k, 64)
# kd = hp.smoothing(k, lmax=128, verbose=False)
#
# plt.clf()
# plt.plot(hp.anafast(kd, lmax=128))
# pixwin = hp.pixwin(64)
# cl = cl[:129] * (pixwin[:129] ** 2)
# plt.plot(cl)
# plt.show()
#
# exit(0)
#
# thetas = np.linspace(0,np.pi/10,1000)
#
# cov_pw = eunomia.sim_tools.covariance.cov_sep_theta_from_cl(thetas, cl_pw)
# cov = eunomia.sim_tools.covariance.cov_sep_theta_from_cl(thetas, cl)
#
# plt.clf()
# plt.plot(thetas, cov/cov[0])
# plt.plot(thetas, cov_pw/cov_pw[0])
# plt.axvline(hp.nside2resol(nside), c='r')
# plt.show()
# exit(0)

# mask = np.load(data_dir + 'mask.npy')
mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))

inds = np.arange(hp.nside2npix(nside))[mask]

# Compute the full covariance matrix for the map from Cl's.
shift = 0.053
# inds = np.arange(10000, dtype=np.int32)
# inds = None
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, inds)
theory_cov = np.log(1 + ln_theory_cov / (shift ** 2))

# mask = ang_sep > 0.5
# theory_cov[mask] = 0

# print(np.linalg.cond(theory_cov))
# print(theory_cov.shape)
# exit(0)

# # Plot the covariance matrix.
# plt.matshow(theory_cov, norm=matplotlib.colors.LogNorm())
# cbar = plt.colorbar()
# plt.suptitle('Full $\kappa$ Covariance')
# plt.savefig(fig_dir + 'kappa_full_cov', dpi=300)
#
# # Show a zoomed in plot of the matrix to show structure.
# plt.matshow(cov[:100, :100], norm=matplotlib.colors.LogNorm())
# plt.colorbar()
# plt.suptitle('Full $\kappa$ Covariance (Zoomed)')
# plt.savefig(fig_dir + 'kappa_full_cov_zoomed', dpi=300)

k2g1, k2g2 = eunomia.sim_tools.shear_conv_transformations.compute_full_conv2shear_mats(nside, lmax, inds)

np.save(out_dir + 'k2g1.npy', k2g1)
np.save(out_dir + 'k2g2.npy', k2g2)

# k2g1 = k2g1[:, mask]
# k2g1 = k2g1[mask, :]
#
# k2g2 = k2g2[:, mask]
# k2g2 = k2g2[mask, :]

g1_obs = g1_obs[mask]
g2_obs = g2_obs[mask]

# sn_std = 0.003
sn_std = 0.0045
sn_var = sn_std ** 2

ms = eunomia.MapSampler(g1_obs, g2_obs, k2g1, k2g2, shift, theory_cov, sn_var, inds)
chain, logp = ms.sample(100, 10, 0, 1.0)

# print(np.linalg.cond(theory_cov))
# print(theory_cov.shape)

plt.clf()
plt.plot(range(len(logp)), logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

plt.clf()
c = ChainConsumer()
c.add_chain(chain[:, :5])
c.plotter.plot(figsize="column")
plt.savefig(fig_dir + 'corner', dpi=300)
