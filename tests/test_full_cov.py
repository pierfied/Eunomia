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
nside = 16
lmax = 2 * nside
npix = hp.nside2npix(nside)

# Load in the convergence map.
kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))
gammas_noise = np.load(data_dir + 'gammas_noise_{0}_{1}.npy'.format(nside, lmax))

k = kappas[:, 0]
kn = kappas_noise[:, 0]
g1_obs = gammas_noise[0, :, 0]
g2_obs = gammas_noise[1, :, 0]

mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True))[:, 1]
pixwin = hp.pixwin(nside)
cl = cl[:lmax + 1] * (pixwin[:lmax + 1] ** 2)

# Compute the full covariance matrix for the map from Cl's.
shift = 0.053
inds = np.arange(npix)[mask]
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, inds)
theory_cov = np.log(1 + ln_theory_cov / (shift ** 2))

np.save(out_dir + 'cov.npy', theory_cov)
# exit(0)

theory_cov = np.load(out_dir + 'cov.npy')

var = theory_cov[0,0]
sigma = np.sqrt(var)
mu = -0.5 * var + np.log(shift)

u, s, vh = np.linalg.svd(theory_cov)

# plt.clf()
# plt.plot(s/s[0])
# plt.savefig(fig_dir + 's.png', dpi=300)
# plt.show()
# exit(0)

rcond = 0.05
good_vecs = s / s[0] > rcond

s = s[good_vecs]
u = u[:, good_vecs]

np.save(out_dir + 'u.npy', u)
np.save(out_dir + 's.npy', s)
# exit(0)

u = np.load(out_dir + 'u.npy')
s = np.load(out_dir + 's.npy')

noise_cov = np.cov(kappas_noise[mask, :])
#
# _, s_noise, _ = np.linalg.svd(noise_cov)
#
# plt.clf()
# plt.plot(s_noise/s_noise[0])
# plt.show()
# exit(0)

inv_noise_cov = np.linalg.pinv(noise_cov, rcond)

ms = eunomia.MapSampler(kn, shift, mu, s, u, inv_noise_cov)
chain, logp = ms.sample(50, 1, 0.5, 10000, 1, 0.5)

# print(np.linalg.cond(theory_cov))
# print(theory_cov.shape)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

chain = chain @ u.T

plt.clf()
plt.plot(range(len(logp)), logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

plt.clf()
c = ChainConsumer()
c.add_chain(chain[:, :5])
c.plotter.plot(figsize="column")
plt.savefig(fig_dir + 'corner', dpi=300)
