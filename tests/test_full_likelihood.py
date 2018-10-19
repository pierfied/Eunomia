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

nside = 16
npix = hp.nside2npix(nside)
lmax = 32

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True, header=None))[:, 1]

pixwin = hp.pixwin(nside)
cl = cl[:len(pixwin)] * (pixwin ** 2)

shift = 0.053

# Compute the full covariance matrix for the map from Cl's.
inds = np.arange(npix)
mask = inds < 1000
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, inds)
theory_cov = np.log(1 + ln_theory_cov/(shift ** 2))

# mask = ang_sep > 0.03

inv_cov = np.linalg.inv(theory_cov)
# inv_cov[mask] = 0

kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))[:,0]

def pyconv2shear(kappa, nside, lmax):
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = hp.ud_grade(kappa, nside)

        k = kappa
        ke = kappa
        kb = np.zeros(hp.nside2npix(nside))

        keb = [k, ke, kb]

        klm, kelm, kblm = hp.map2alm(keb, pol=False, lmax=lmax)

        l, _ = hp.Alm.getlm(hp.Alm.getlmax(len(klm)))

        gelm = np.zeros(len(l), dtype=np.complex128)
        gblm = np.zeros(len(l), dtype=np.complex128)

        gelm = -np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kelm
        gblm = -np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kblm
        gelm[l == 0] = 0
        gblm[l == 0] = 0

        glm = gelm + 1j * gblm
        geblm = [glm, gelm, gblm]

        gqu = np.array(hp.alm2map(geblm, nside=nside, lmax=lmax, verbose=False))

        return gqu

def pyshear2conv(gqu, nside, lmax):
    with np.errstate(divide='ignore', invalid='ignore'):
        gqu = hp.ud_grade(gqu, nside)

        glm, gelm, gblm = hp.map2alm(gqu, lmax=lmax)

        l, _ = hp.Alm.getlm(hp.Alm.getlmax(len(glm)))

        kelm = np.zeros(len(l), dtype=np.complex128)
        kblm = np.zeros(len(l), dtype=np.complex128)

        kelm = -np.sqrt(l * (l + 1) / ((l + 2) * (l - 1))) * gelm
        kblm = -np.sqrt(l * (l + 1) / ((l + 2) * (l - 1))) * gblm
        kelm[l == 0] = 0
        kblm[l == 0] = 0
        kelm[l == 1] = 0
        kblm[l == 1] = 0

        klm = kelm + 1j * kblm
        keblm = [klm, kelm, kblm]

        k, ke, kb = hp.alm2map(keblm, nside=nside, pol=False, lmax=lmax, verbose=False)

        return ke

kappas = hp.smoothing(kappas, lmax=lmax, verbose=False)

shears = pyconv2shear(kappas, nside, lmax)
shape_noise_var = (shears.std() * 2) ** 2
shears_w_noise = shears + np.random.standard_normal(shears.shape) * np.sqrt(shape_noise_var)
shears_w_noise_and_masking = shears_w_noise.copy()
shears_w_noise_and_masking[:,~mask] = 0
shears_obs = shears_w_noise_and_masking.copy()

kappas_obs = pyshear2conv(shears_obs, nside, lmax)
kappas_obs[~mask] = 0
kappas_obs[mask] -= kappas_obs[mask].mean()

kappa_noise_var = (pyshear2conv(shears_w_noise, nside, lmax) - kappas).var()
resid_cov = np.eye(len(inds)) * kappa_noise_var

k_true = kappas.copy()
k_obs = kappas_obs.copy()

np.save(out_dir + 'k_true.npy', k_true)
np.save(out_dir + 'k_obs.npy', k_obs)
np.save(out_dir + 'mask.npy', mask)

y_true = np.log(shift + k_true)
y_obs = np.log(shift + k_obs)

masked_inds = np.zeros_like(mask, dtype=np.int32)
masked_inds[mask] = 1
masked_inds[~mask] = 0

shape_noise_1 = np.ones_like(k_true) * shape_noise_var
shape_noise_2 = np.ones_like(k_true) * shape_noise_var

theory_cov *= 0
inv_cov *= 0

ms = eunomia.MapSampler(shears_obs[1,:], shears_obs[2,:], k_obs, theory_cov, inv_cov, shift, masked_inds, nside, lmax, shape_noise_1, shape_noise_2, resid_cov)
chain, logp = ms.sample(1000, 10, 500, 1.0)

plt.plot(range(len(logp)),logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)
plt.clf()

np.save(out_dir + 'no_prior_chain.npy', chain)
np.save(out_dir + 'no_prior_logp.npy', logp)

# k_obs_samps = np.stack([np.random.normal(loc=ko, scale=np.sqrt(resid_var), size=10000) for ko in k_obs], axis=1)
# y_obs_samps = np.log(shift + k_obs_samps)

# y_obs_samps = np.stack([np.random.normal(loc=yo, scale=np.sqrt(resid_var), size=10000) for yo in y_obs], axis=1)
#
# print(chain.shape)
#
# c = ChainConsumer()
# samp_inds = np.random.choice(inds, 5)
# c.add_chain(np.exp(y_obs_samps[:,samp_inds]) - shift, name='Likelihood', parameters=['1','2','3','4','5'])
# c.add_chain(np.exp(chain[:,samp_inds]) - shift, name='Likelihood $\\times$ Prior', parameters=['1','2','3','4','5'])
# c.plotter.plot(figsize='column', truth=k_true[samp_inds])
# plt.suptitle('$\\kappa$ Pixel Samples')
# plt.savefig(fig_dir + 'corner.png', bbox_inches='tight')
#
# c = ChainConsumer()
# c.add_chain(chain)
# y_new = [param[1] for param in c.analysis.get_summary().values()]
# print(y_new)
#
# plt.clf()
# _, bins, _ = plt.hist(y_obs - y_true, 50)
# plt.hist(y_new - y_true, bins)
# plt.savefig(fig_dir + 'y_diff_hist.png', dpi=300)