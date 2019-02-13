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
nside = 32
lmax = 2 * nside
npix = hp.nside2npix(nside)

mask = np.load(data_dir + 'des_y1_mask_{0}.npy'.format(nside))

# Load in the convergence map.
kappas = np.load(data_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(data_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))
gammas_noise = np.load(data_dir + 'gammas_noise_{0}_{1}.npy'.format(nside, lmax))

# for i in tqdm(range(kappas.shape[1])):
#     kappas[mask, :] -= kappas[mask, :].mean()
#     kappas_noise[mask, :] -= kappas_noise[mask, :].mean()
#
# np.save(out_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax), kappas)
# np.save(out_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax), kappas_noise)
# exit(0)

kappas = np.load(out_dir + 'kappas_{0}_{1}.npy'.format(nside, lmax))
kappas_noise = np.load(out_dir + 'kappas_noise_{0}_{1}.npy'.format(nside, lmax))

k = kappas[:, 2]
kn = kappas_noise[:, 2]
g1_obs = gammas_noise[0, :, 2]
g2_obs = gammas_noise[1, :, 2]

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

################################################################################

theory_cov = np.load(out_dir + 'cov.npy')

var = theory_cov[0, 0]
sigma = np.sqrt(var)
mu = -0.5 * var + np.log(shift)

u, s, vh = np.linalg.svd(theory_cov)

# plt.clf()
# plt.plot(s/s[0])
# plt.show()
# exit(0)
#
rcond = 0.2
good_vecs = s / s[0] > rcond

s = s[good_vecs]
u = u[:, good_vecs]

theory_cov = u @ np.diag(s) @ u.T
inv_theory_cov = u @ np.diag(1 / s) @ u.T
#
# np.save(out_dir + 'u.npy', u)
# np.save(out_dir + 's.npy', s)
# # exit(0)
#
# u = np.load(out_dir + 'u.npy')
# s = np.load(out_dir + 's.npy')

################################################################################

noise_cov = np.cov(kappas_noise[mask, :] - kappas[mask, :])

u_k, s_k, vh_k = np.linalg.svd(noise_cov)

# plt.plot(s_k/s_k[0])
# plt.show()
# exit(0)

k_cond = 0.01
good_vecs = s_k / s_k[0] > k_cond

s_k = s_k[good_vecs]
u_k = u_k[:, good_vecs]

noise_cov = u_k @ np.diag(s_k) @ u_k.T
inv_noise_cov = u_k @ np.diag(1 / s_k) @ u_k.T

################################################################################

y_true = np.log(shift + kappas[mask, :])
y_noise = np.log(shift + kappas_noise[mask, :])

y_noise_cov = np.cov(y_noise - y_true)

u_y, s_y, vh_y = np.linalg.svd(y_noise_cov)

# plt.semilogy(s_y/s_y[0])
# plt.show()
# exit(0)

y_cond = 0.001
good_vecs = s_y / s_y[0] > y_cond

s_y = s_y[good_vecs]
u_y = u_y[:, good_vecs]

y_noise_cov = u_y @ np.diag(s_y) @ u_y.T

################################################################################

u_sum, s_sum, vh_sum = np.linalg.svd(theory_cov + y_noise_cov)

# plt.semilogy(s_sum/s_sum[0])
# plt.show()
# exit(0)

sum_cond = 0.0005
good_vecs = s_sum / s_sum[0] > sum_cond

s_sum = s_sum[good_vecs]
u_sum = u_sum[:, good_vecs]

inv_sum = u_sum @ np.diag(1 / s_sum) @ u_sum.T

################################################################################

joint_cov = theory_cov @ inv_sum @ y_noise_cov

u_j, s_j, v_j = np.linalg.svd(joint_cov)

# plt.plot(s_j / s_j[0])
# plt.show()
# exit(0)

j_cond = 0.15
good_vecs = s_j / s_j[0] > j_cond

s_j = s_j[good_vecs]
u_j = u_j[:, good_vecs]

mu_j = y_noise_cov @ inv_sum @ np.ones((mask.sum(), 1)) * mu + theory_cov @ inv_sum @ y_noise[:, 2:3]

################################################################################

# plt.figure()
# plt.imshow(u_j.T @ theory_cov @ u_j)
# plt.figure()
# plt.imshow(u_j.T @ y_noise_cov @ u_j)
# plt.show()
# exit(0)

################################################################################

# plt.figure()
# plt.imshow(inv_theory_cov)
# plt.figure()
# plt.imshow(inv_noise_cov)
# plt.show()
# exit(0)

# print(kn.shape)
# print(mask.sum())
# print(mu_j.ravel().shape)
# exit(0)

# print(inv_theory_cov)

ms = eunomia.MapSampler(kn[mask], s_j, u_j, mu_j, shift, mu, inv_theory_cov, inv_noise_cov)
chain, logp = ms.sample(50, 1, 0.5, 10000, 1, 0.2)
# exit(0)

ms = eunomia.MapSampler(kn, s, u, mu * mu_j / mu_j, shift, mu, inv_theory_cov, inv_noise_cov * 0)
prior_chain, logp = ms.sample(50, 1, 0.5, 10000, 1, 0.5)

ms = eunomia.MapSampler(kn[mask], s_y, u_y, y_noise[:, 2:3], shift, mu, inv_theory_cov * 0, inv_noise_cov)
noise_chain, logp = ms.sample(50, 1, 0.5, 10000, 1, 0.5)

# print(np.linalg.cond(theory_cov))
# print(theory_cov.shape)

np.save(out_dir + 'chain.npy', chain)
np.save(out_dir + 'logp.npy', logp)

chain = np.exp(chain @ u_j.T + np.tile(mu_j.T, (10000, 1))) - shift
prior_chain = np.exp(prior_chain @ u.T + mu) - shift
noise_chain = np.exp(noise_chain @ u_y.T + np.tile(y_noise[:, 2:3].T, (10000, 1))) - shift

for i in range(chain.shape[0]):
    chain[i, :] -= chain[i, :].mean()
    prior_chain[i, :] -= prior_chain[i, :].mean()
    noise_chain[i, :] -= noise_chain[i, :].mean()

plt.clf()
plt.plot(range(len(logp)), logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300)

k[mask] -= k[mask].mean()
kn[mask] -= kn[mask].mean()
ki = chain.mean(axis=0)

plt.clf()
_, bins, _ = plt.hist(k[mask] - kn[mask], 10, label='Observed', alpha=0.5)
plt.hist(k[mask] - ki, bins, label='Improved', alpha=0.5)
plt.xlabel('$\\Delta \\kappa_i$')
plt.ylabel('Number of Pixels')
plt.legend()
plt.savefig(fig_dir + 'dk_comp.png', dpi=300, bbox_inches='tight')

plot_inds = np.random.choice(np.arange(mask.sum()), 5, replace=False)

plt.clf()
c = ChainConsumer()
c.add_chain(prior_chain[:, plot_inds], name='Prior')
c.add_chain(noise_chain[:, plot_inds], name='Noise')
c.add_chain(chain[:, plot_inds], name='Noise + Prior')
c.plotter.plot(figsize="column", truth=k[mask][plot_inds])
plt.savefig(fig_dir + 'corner', dpi=300, bbox_inches='tight')
