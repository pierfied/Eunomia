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
import matplotlib as mpl

# Set important directories and create the figure directory if necessary.
data_dir = './test_data/'
fig_dir = './test_figs/'
out_dir = './test_out/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Determine nside and lmax.
nside = 32
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
# # g1_obs, g2_obs = eunomia.sim_tools.shear_conv_transformations.conv2shear(k, lmax)
g1_obs = gammas_noise[0, :, 0]
g2_obs = gammas_noise[1, :, 0]

# kn = k.copy()
# g1_obs = k.copy()
# g2_obs = k.copy()

# Plot the convergence map.
# hp.mollview(kn, title='Flask Sim Convergence Map $n_{side}=%d$, $\ell_{max}=%d$' % (nside, lmax), unit='$\kappa$')
# plt.savefig(fig_dir + 'kappa_map', dpi=300)

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True))[:, 1]
pixwin = hp.pixwin(nside)
# cl_pw = cl[:len(pixwin)] * (pixwin ** 2)
cl = cl[:lmax + 1] * (pixwin[:lmax + 1] ** 2)

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
# mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))
#
# inds = np.arange(hp.nside2npix(nside))[mask]

buffer_size = 5
bmask = np.load(data_dir + 'bmask_{0}_{1}.npy'.format(nside, buffer_size))
mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))

################################################################################

# Compute the full covariance matrix for the map from Cl's.
shift = 0.053
# inds = np.arange(10000, dtype=np.int32)
# inds = None
inds = np.arange(hp.nside2npix(nside))[bmask]
# inds = np.arange(10000)
# ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, inds)
# theory_cov = np.log(1 + ln_theory_cov / (shift ** 2))
#
# np.save(out_dir + 'cov.npy', theory_cov)
# exit(0)

theory_cov = np.load(out_dir + 'cov.npy')

var = theory_cov[0,0]
sigma = np.sqrt(var)
mu = -0.5 * var + np.log(shift)

################################################################################

# u, s, vh = np.linalg.svd(theory_cov)
#
# # plt.clf()
# # plt.semilogy(s/s[0])
# # # plt.savefig(fig_dir + 's.png', dpi=300)
# # plt.show()
# # exit(0)
#
# rcond = 0.2
# # rcond = 0.6
# good_vecs = s / s[0] > rcond
#
# s = s[good_vecs]
# u = u[:, good_vecs]
#
# # s = s[:500]
# # u = u[:,:500]
#
# np.save(out_dir + 'u.npy', u)
# np.save(out_dir + 's.npy', s)
# # exit(0)

u = np.load(out_dir + 'u.npy')
s = np.load(out_dir + 's.npy')

################################################################################

# k2g1, k2g2 = eunomia.sim_tools.shear_conv_transformations.compute_full_conv2shear_mats(nside, lmax, mask, bmask)
#
# np.save(out_dir + 'k2g1.npy', k2g1)
# np.save(out_dir + 'k2g2.npy', k2g2)
# exit(0)

k2g1 = np.load(out_dir + 'k2g1.npy')
k2g2 = np.load(out_dir + 'k2g2.npy')

################################################################################

sn_std = 0.0045

S = np.ones(mask.sum()) * (sn_std ** 2)

g1_obs = g1_obs[mask]
g2_obs = g2_obs[mask]

# sn_std = 0.003
# sn_std = 0.0014
# sn_std = 0.0045
# # sn_std = 0.009
# sn_var = sn_std ** 2

# sn_var = np.load(data_dir + 'shape_noise.npy')

# theory_cov = np.load(out_dir + 'cov.npy')
#
# u,s,v = np.linalg.svd(theory_cov)
#
# tcond = 0.2
#
# good_vecs = s/s[0] > tcond
#
# s = s[good_vecs]
# u = u[:, good_vecs]
#
# # theory_cov = u @ np.diag(s) @ u.T
theory_inv = u @ np.diag(1/s) @ u.T

################################################################################

ms = eunomia.MapSampler(g1_obs, g2_obs, k2g1, k2g2, shift, mu, s, u, np.ones(u.shape[0]) * mu, theory_inv, S, inds)
chain, logp = ms.sample(1000, 1, 0.5, 1e-2)
# exit(0)

# print(np.linalg.cond(theory_cov))
# print(theory_cov.shape)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

mask_in_bmask = np.zeros_like(mask, dtype=bool)
mask_in_bmask[mask] = True
mask_in_bmask = mask_in_bmask[bmask]

chain = chain @ u.T

for i in tqdm(range(chain.shape[0])):
    chain[i,:] += mu

chain = (np.exp(chain) - shift)[:,mask_in_bmask]

for i in tqdm(range(chain.shape[0])):
    chain[i,:] -= chain[i,:].mean()

k_true = k[mask] - k[mask].mean()
k_noise = kn[mask] - kn[mask].mean()
k_chain = chain[-1,:]

m = np.zeros(hp.nside2npix(nside))
m[mask] = k_chain
cl_chain = hp.anafast(m)
plt.clf()
hp.orthview(m, rot=[30,-65], half_sky=True, title='HMC-SA Max Likelihood $\\kappa$ Map', cbar=None)
hp.graticule(10,10)
ax = plt.gca()
image = ax.get_images()[0]
plt.colorbar(image)
image.norm = mpl.colors.SymLogNorm(0.001)
image.set_clim(-0.01,0.01)
plt.savefig(fig_dir + 'sa_map.png', dpi=300, bbox_inches='tight')

m[mask] = k_true
cl_true = hp.anafast(m)
plt.clf()
hp.orthview(m, rot=[30,-65], half_sky=True, title='True $\\kappa$ Map', cbar=None)
hp.graticule(10,10)
ax = plt.gca()
image = ax.get_images()[0]
plt.colorbar(image)
image.norm = mpl.colors.SymLogNorm(0.001)
image.set_clim(-0.01,0.01)
plt.savefig(fig_dir + 'true_map.png', dpi=300, bbox_inches='tight')

m[mask] = k_noise
cl_noise = hp.anafast(m)
plt.clf()
hp.orthview(m, rot=[30,-65], half_sky=True, title='KSB $\\kappa$ Map', cbar=None)
hp.graticule(10,10)
ax = plt.gca()
image = ax.get_images()[0]
plt.colorbar(image)
image.norm = mpl.colors.SymLogNorm(0.001)
image.set_clim(-0.01,0.01)
plt.savefig(fig_dir + 'ksb_map.png', dpi=300, bbox_inches='tight')

delta_k_noise = k_noise - k_true
delta_k_chain = k_chain - k_true

plt.clf()
_, bins = np.histogram(np.concatenate((delta_k_chain, delta_k_noise)), 10)
plt.hist(k_noise - k_true, bins, alpha=0.5, label='KSB')
plt.hist(k_chain - k_true, bins, alpha=0.5, label='HMC-SA Max Likelihood')
plt.xlabel('$\\Delta \\kappa$')
plt.ylabel('Number of Pixels')
plt.legend()
plt.savefig(fig_dir + 'dk_hist.png', dpi=300, bbox_inches='tight')

plt.clf()
plt.plot(cl_true, label='True')
plt.plot(cl_noise, label='KSB')
plt.plot(cl_chain, label='HMC-SA Max Likelihood')
plt.xlabel('$\\ell$')
plt.ylabel('$C_\\ell$')
plt.legend()
plt.savefig(fig_dir + 'cl_comp.png', dpi=300, bbox_inches='tight')

plt.clf()
plt.semilogy(range(len(logp)), -logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

exit(0)

plt.clf()
c = ChainConsumer()
c.add_chain(chain[:, :5])
c.plotter.plot(figsize="column", truth=k_true[:5])
plt.savefig(fig_dir + 'corner', dpi=300)
