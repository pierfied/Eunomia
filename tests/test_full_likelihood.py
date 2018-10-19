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

kappa_nside = 16
gamma_nside = 4
kappa_npix = hp.nside2npix(kappa_nside)
gamma_npix = hp.nside2npix(gamma_nside)
lmax = 16

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True, header=None))[:, 1]

pixwin = hp.pixwin(kappa_nside)
cl = cl[:len(pixwin)] * (pixwin ** 2)

shift = 0.053

# Compute the full covariance matrix for the map from Cl's.
gamma_inds = np.arange(gamma_npix)
kappa_inds = np.arange(kappa_npix)
gamma_mask = gamma_inds > 0
kappa_mask = hp.ud_grade(gamma_mask, kappa_nside)
ln_theory_cov, ang_sep = eunomia.sim_tools.covariance.full_cov_from_cl(cl, kappa_nside, kappa_inds)
theory_cov = np.log(1 + ln_theory_cov/(shift ** 2))

# mask = ang_sep > 0.03

inv_cov = np.linalg.inv(theory_cov)
# inv_cov[mask] = 0

kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(kappa_nside, lmax))[:,0]

def pyconv2shear(kappa, nside, lmax):
    nside = hp.npix2nside(len(kappa))

    k = kappa
    ke = kappa
    kb = np.zeros(len(kappa))

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

    gqu = np.array(hp.alm2map(geblm, nside=nside, lmax=lmax))

    return gqu

def pyshear2conv(gqu, nside, lmax):
    nside = hp.npix2nside(gqu.shape[1])

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

    k, ke, kb = hp.alm2map(keblm, nside=nside, pol=False, lmax=lmax)

    return ke, kb

high_res_shears = pyconv2shear(kappas, kappa_nside, lmax)
high_res_shears_plus_noise = high_res_shears + np.random.standard_normal(high_res_shears.shape) * high_res_shears.std() * 3

shears = hp.ud_grade(high_res_shears, gamma_nside)
shears_plus_noise = hp.ud_grade(high_res_shears_plus_noise, gamma_nside)

low_res_kappas = hp.ud_grade(kappas, gamma_nside)
recov_kappas = hp.ud_grade(pyshear2conv(hp.ud_grade(shears, kappa_nside), kappa_nside, None)[0], gamma_nside)

plt.hist((recov_kappas - low_res_kappas)/low_res_kappas.std(),20)
plt.show()
exit(0)





kappas_plus_noise = pyshear2conv(high_res_shears_plus_noise, kappa_nside, lmax)[0]
kappas_plus_noise = hp.ud_grade(kappas_plus_noise, gamma_nside)
kappa_d = hp.ud_grade(kappas, gamma_nside)

kappa_noise_d = kappas_plus_noise - kappa_d
kappa_noise_var =  kappa_nside/gamma_nside * kappa_noise_d.var()

high_res_shears = hp.ud_grade(high_res_shears, gamma_nside)
high_res_shears_plus_noise = hp.ud_grade(high_res_shears_plus_noise, gamma_nside)

shear_noise = high_res_shears_plus_noise - high_res_shears
shear_noise_var = shear_noise.var()

resid_cov = np.eye(len(inds)) * kappa_noise_var

k_true = kappas
k_obs = hp.ud_grade(hp.ud_grade(kappas_plus_noise, gamma_nside), kappa_nside)

y_true = np.log(shift + k_true)
y_obs = np.log(shift + k_obs)

ms = eunomia.MapSampler(y_obs, theory_cov, inv_cov, shift, resid_cov)
chain, logp = ms.sample(1000, 10, 100, 1.0)

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