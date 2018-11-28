from astropy.io import fits
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import eunomia
import healpy as hp
import pandas as pd
import os
from chainconsumer import ChainConsumer
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import pickle
from tqdm import tqdm

# Set important directories and create the figure directory if necessary.
data_dir = './test_data/'
fig_dir = './test_figs/'
out_dir = './test_out/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

nside = 128
npix = hp.nside2npix(nside)
lmax = 256

# Load in the harmonic covariance values (Cl).
cl = np.array(pd.read_csv(data_dir + 'full_cl.dat', delim_whitespace=True, header=None))[:, 1]

pixwin = hp.pixwin(nside)
cl = cl[:len(pixwin)] * (pixwin ** 2)

cl = cl[:lmax + 1]

# thetas = np.linspace(0, np.pi/80, 1000)
#
# covs = eunomia.sim_tools.covariance.cov_sep_theta_from_cl(thetas, cl)
#
# resol = hp.nside2resol(nside)
#
# plt.plot(np.rad2deg(thetas), covs)
# plt.axvline(resol, c='r')
# plt.show()
# exit(0)

shift = 0.053

max_sep = np.pi/80

# Compute the full covariance matrix for the map from Cl's.
inds = np.arange(npix)
# mask = inds < 1000
# ln_theory_cov = eunomia.sim_tools.covariance.full_cov_from_cl(cl, nside, max_sep)
#
# theory_cov = ln_theory_cov.copy()
# theory_cov.data = np.log(1 + ln_theory_cov.data / (shift ** 2))
#
# # np.save(out_dir + 'theory_cov_{0}_{1}.npy'.format(nside, lmax), theory_cov)
#
# theory_cov_file = open(out_dir + 'theory_cov_{0}_{1}.npy'.format(nside, lmax), 'wb')
# pickle.dump(theory_cov, theory_cov_file)
# theory_cov_file.close()
#
# exit(0)

# theory_cov = np.load(out_dir + 'theory_cov_{0}_{1}.npy'.format(nside, lmax))

theory_cov_file = open(out_dir + 'theory_cov_{0}_{1}.npy'.format(nside, lmax), 'rb')
theory_cov = pickle.load(theory_cov_file)
theory_cov_file.close()

# print(type(theory_cov))
# print(theory_cov)
# exit(0)

mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))

y_mu = np.log(shift) - 0.5 * theory_cov[0, 0]

# mask = ang_sep > 0.03

# inv_cov = np.linalg.inv(theory_cov)
# inv_cov[mask] = 0

kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))
# mask = np.load(data_dir + 'mask.npy')

y = np.log(shift + kappas)
y_noise = np.log(shift + kappas_noise)

# resid = (y_noise - y)[mask]
# full_resid_cov = np.cov(resid)
# resid_cov = np.zeros_like(full_resid_cov)
#
# pix_vecs = np.array(hp.pix2vec(nside, inds[mask]))
#
# for i in tqdm(range(mask.sum())):
#     neighbor_seps =  hp.rotator.angdist(pix_vecs[:, i], pix_vecs)
#
#     good_neighbors = neighbor_seps < max_sep
#
#     resid_cov[i, good_neighbors] = full_resid_cov[i, good_neighbors]

rcond = 1e-4

# print('Inverting Residual Covariance')
# inv_resid_cov = np.linalg.pinv(resid_cov, rcond=rcond)
# print('Uninverting Residual Covariance')
# resid_cov = np.linalg.pinv(inv_resid_cov, rcond=rcond)
#
# full_inv_resid_cov = sp.lil_matrix((npix, npix))
# print('Building Sparse Residual Covariance')
# for i in tqdm(range(mask.sum())):
#     full_inv_resid_cov[inds[mask][i], inds[mask]] = resid_cov[i,:]
#
# full_inv_resid_cov = sp.csr_matrix(full_inv_resid_cov)
# full_inv_resid_cov.eliminate_zeros()
#
# full_inv_resid_cov_file = open(out_dir + 'full_inv_resid_cov_{0}_{1}.pkl'.format(nside, lmax), 'wb')
# full_inv_resid_cov = pickle.dump(full_inv_resid_cov, full_inv_resid_cov_file)
# full_inv_resid_cov_file.close()

full_inv_resid_cov_file = open(out_dir + 'full_inv_resid_cov_{0}_{1}.pkl'.format(nside, lmax), 'rb')
full_inv_resid_cov = pickle.load(full_inv_resid_cov_file)
full_inv_resid_cov_file.close()

# print(full_inv_resid_cov)
# exit(0)

print('Inverting Sparse Theory Covariance')

# inv_theory_cov = np.linalg.pinv(theory_cov, rcond=rcond)
k = 1000
s = spl.svds(theory_cov, k=k, return_singular_vectors=False)

print(s)
exit(0)

inv_cov = inv_theory_cov + full_inv_resid_cov
cov = np.linalg.pinv(inv_cov, rcond=rcond)
inv_cov = np.linalg.pinv(cov, rcond=rcond)

k_true = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))[:, 0]
k_obs = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))[:, 0]

y_true = np.log(shift + k_true)
y_obs = np.log(shift + k_obs)

mu = cov @ (inv_theory_cov @ (y_mu * np.ones(npix)) + full_inv_resid_cov @ y_obs)


def conv2shear(kappa, nside, lmax=None):
    kelm = hp.map2alm(kappa, lmax=lmax)

    l, _ = hp.Alm.getlm(hp.Alm.getlmax(len(kelm)))

    gelm = -np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kelm
    gelm[l == 0] = 0

    gblm = np.zeros_like(gelm)
    glm = gelm

    _, g1, g2 = hp.alm2map([glm, gelm, gblm], nside=nside, lmax=lmax, verbose=False)

    return g1, g2


def shear2conv(gamma1, gamma2, nside, lmax=None):
    g = gamma1 + 1j * gamma2

    glm, gelm, gblm = hp.map2alm([g, gamma1, gamma2], lmax=lmax)

    l, _ = hp.Alm.getlm(hp.Alm.getlmax(len(glm)))

    factor = -np.sqrt(l * (l + 1) / ((l + 2) * (l - 1)))

    kelm = factor * gelm
    kelm[l == 0] = 0
    kelm[l == 1] = 0

    kblm = factor * gblm
    kblm[l == 0] = 0
    kblm[l == 1] = 0

    klm = kelm + 1j * kblm

    k, ke, kb = hp.alm2map([klm, kelm, kblm], nside=nside, pol=False, lmax=lmax, verbose=False)

    return ke


gammas_obs = np.load(data_dir + 'gammas_noise_{0}_{1}.npy'.format(nside, lmax))
g1_obs = gammas_obs[0,:,0]
g2_obs = gammas_obs[1,:,0]

sn_std = 0.003 * (nside / 16)
sn_var = sn_std ** 2
sn_inv_cov = np.eye(mask.sum()) / sn_var


def sample_resid_kappa_likelihood(y_obs, u, sqrt_s, shift, mask):
    np.random.seed()

    samp_diag_vec = sqrt_s * np.random.standard_normal(mask.sum())

    samp_y = y_obs[mask] + (u @ samp_diag_vec)

    samp_kappa_mask = np.exp(samp_y) - shift

    samp_kappa = np.zeros(len(mask))
    samp_kappa[mask] = samp_kappa_mask

    return samp_kappa


def sample_full_kappa_likelihood(mu, u, sqrt_s, shift, npix):
    np.random.seed()

    samp_diag_vec = sqrt_s * np.random.standard_normal(npix)

    samp_y = mu + (u @ samp_diag_vec)

    samp_kappa = np.exp(samp_y) - shift

    return samp_kappa

# nsamps = 1000
#
# print('Sampling w/ Prior Only')
#
# u, s, _ = np.linalg.svd(resid_cov)
# sqrt_s = np.sqrt(s)
# no_prior_samples = np.stack([sample_resid_kappa_likelihood(y_obs, u, sqrt_s, shift, mask) for i in range(nsamps)])
#
# print('Sampling Full Likelihood')
#
# u, s, _ = np.linalg.svd(cov)
# sqrt_s = np.sqrt(s)
# full_samples = np.stack([sample_full_kappa_likelihood(mu, u, sqrt_s, shift, npix) for i in range(nsamps)])
#
# c = ChainConsumer()
# c.add_chain(no_prior_samples[:, :5])
# c.add_chain(no_prior_samples[:, :5])
# c.plotter.plot(figsize='column', truth=k_true[:5])
# plt.show()


nsims = kappas.shape[1]

avg_map = np.zeros(npix)
avg_updated_map = np.zeros(npix)
rms_map = np.zeros(npix)

print('Looping over map realizations.')

for i in range(nsims):
    print(i)

    k_true = kappas[:,i]
    k_obs = kappas_noise[:,i]

    k_true[~mask] = 0
    k_true[mask] -= k_true[mask].mean()

    k_obs[~mask] = 0
    k_obs[mask] -= k_obs[mask].mean()

    y_true = np.log(shift + k_true)
    y_obs = np.log(shift + k_obs)

    new_y = cov @ (inv_theory_cov @ (y_mu * np.ones(npix)) + full_inv_resid_cov @ y_obs)

    new_kappa = np.exp(new_y) - shift

    new_kappa[~mask] = 0
    new_kappa[mask] -= new_kappa[mask].mean()

    delta_kappa_obs = k_obs - k_true
    delta_kappa_updated = new_kappa - k_true

    avg_map += delta_kappa_obs
    avg_updated_map += delta_kappa_updated


avg_map[~mask] = np.nan
avg_updated_map[~mask] = np.nan

avg_map /= nsims
avg_updated_map /= nsims

rms_map = np.zeros(npix)
rms_updated_map = np.zeros(npix)

for i in range(nsims):
    print(i)

    k_true = kappas[:,i]
    k_obs = kappas_noise[:,i]

    k_true[~mask] = 0
    k_true[mask] -= k_true[mask].mean()

    k_obs[~mask] = 0
    k_obs[mask] -= k_obs[mask].mean()

    y_true = np.log(shift + k_true)
    y_obs = np.log(shift + k_obs)

    new_y = cov @ (inv_theory_cov @ (y_mu * np.ones(npix)) + full_inv_resid_cov @ y_obs)

    new_kappa = np.exp(new_y) - shift

    new_kappa[~mask] = 0
    new_kappa[mask] -= new_kappa[mask].mean()

    delta_kappa_obs = k_obs - k_true
    delta_kappa_updated = new_kappa - k_true

    rms_map += (delta_kappa_obs - avg_map) ** 2
    rms_updated_map += (delta_kappa_updated - avg_updated_map) ** 2

rms_map = np.sqrt(rms_map / nsims)
rms_updated_map = np.sqrt(rms_updated_map / nsims)

rms_map[~mask] = np.nan
rms_updated_map[~mask] = np.nan

hp.mollview(avg_map, min=-0.00025, max=0.00025, title='$\\langle \\Delta \\kappa_{\\mathrm{KSB}} \\rangle = \\langle \\kappa_{\\mathrm{KSB}} - \\kappa_{\\mathrm{true}} \\rangle$')
plt.savefig(fig_dir + 'dk_ksb.png', dpi=300, bbox_inches='tight')
hp.mollview(avg_updated_map, min=-0.00025, max=0.00025, title='$\\langle \\Delta \\widetilde{\\kappa} \\rangle = \\langle \\widetilde{\\kappa} - \\kappa_{\\mathrm{true}} \\rangle$')
plt.savefig(fig_dir + 'dk_ln.png', dpi=300, bbox_inches='tight')
plt.show()

hp.mollview(rms_map, min=0.001, max=0.002, title='RMS $\\Delta \\kappa_{\\mathrm{KSB}}$')
plt.savefig(fig_dir + 'rms_dk_ksb.png', dpi=300, bbox_inches='tight')
hp.mollview(rms_updated_map, min=0.001, max=0.002, title='RMS $\\Delta \\widetilde{\\kappa}$')
plt.savefig(fig_dir + 'rms_dk_ln.png', dpi=300, bbox_inches='tight')
plt.show()

plt.hist(avg_map, 20, alpha=0.5, density=True, label='KSB')
plt.hist(avg_updated_map, 20, alpha=0.5, density=True, label='$\\widetilde{\\kappa}$')
plt.xlabel('$\\Delta \\kappa$')
plt.ylabel('Number of Pixels')
plt.legend()
plt.title('Difference Histogram')
plt.savefig(fig_dir + 'hist_dk.png', dpi=300, bbox_inches='tight')

plt.figure()

_,bins,_ = plt.hist(rms_map, 20, alpha=0.5, density=True, label='KSB')
plt.hist(rms_updated_map, 20, alpha=0.5, density=True, label='$\\widetilde{\\kappa}$')
plt.xlabel('RMS $\\Delta \\kappa$')
plt.ylabel('Number of Pixels')
plt.legend()
plt.title('RMS Difference Histogram')
plt.savefig(fig_dir + 'hist_rms_dk.png', dpi=300, bbox_inches='tight')
plt.show()