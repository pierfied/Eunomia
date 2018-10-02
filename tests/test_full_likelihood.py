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
nside = 128
npix = hp.nside2npix(nside)
lmax = 4 * nside

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True, header=None))[:, 1]

pixwin = hp.pixwin(nside)
cl = cl[:len(pixwin)] * (pixwin ** 2)

shift = 0.053

# Compute the full covariance matrix for the map from Cl's.
inds = np.arange(1000)
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, inds)
theory_cov = np.log(1 + ln_theory_cov/(shift ** 2))

# mask = ang_sep > 0.03

inv_cov = np.linalg.inv(theory_cov)
# inv_cov[mask] = 0

kappas = np.load(data_dir + 'kappas_128_512.npy')
kappas_noise = np.load(data_dir + 'kappas_noise_128_512.npy')

# resid_var = np.var(np.log(1 + kappas_noise) - np.log(1 + kappas))
# resid_var = np.var(kappas_noise - kappas)
resid_var = np.var(np.log(shift + kappas_noise) - np.log(shift + kappas))

resid_cov = np.eye(len(inds)) * resid_var

k_true = kappas[inds, 0]
k_obs = kappas_noise[inds, 0]

y_true = np.log(shift + k_true)
y_obs = np.log(shift + k_obs)

ms = eunomia.MapSampler(y_obs, theory_cov, inv_cov, shift, resid_cov)
chain, logp = ms.sample(10000, 10, 100, 1.0)

plt.plot(range(len(logp)),logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)
plt.clf()

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

# k_obs_samps = np.stack([np.random.normal(loc=ko, scale=np.sqrt(resid_var), size=10000) for ko in k_obs], axis=1)
# y_obs_samps = np.log(shift + k_obs_samps)

y_obs_samps = np.stack([np.random.normal(loc=yo, scale=np.sqrt(resid_var), size=10000) for yo in y_obs], axis=1)

print(chain.shape)

c = ChainConsumer()
samp_inds = np.random.choice(inds, 5)
c.add_chain(np.exp(y_obs_samps[:,samp_inds]) - shift, name='Likelihood', parameters=['1','2','3','4','5'])
c.add_chain(np.exp(chain[:,samp_inds]) - shift, name='Likelihood $\\times$ Prior', parameters=['1','2','3','4','5'])
c.plotter.plot(figsize='column', truth=k_true[samp_inds])
plt.suptitle('$\\kappa$ Pixel Samples')
plt.savefig(fig_dir + 'corner.png', bbox_inches='tight')

c = ChainConsumer()
c.add_chain(chain)
y_new = [param[1] for param in c.analysis.get_summary().values()]
print(y_new)

plt.clf()
_, bins, _ = plt.hist(y_obs - y_true, 50)
plt.hist(y_new - y_true, bins)
plt.savefig(fig_dir + 'y_diff_hist.png', dpi=300)