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
shift = 0.053

inds = np.arange(10000)
mask = np.zeros(hp.nside2npix(nside), dtype=bool)
mask[inds] = True

nsims = kappas.shape[1]
# nsims = 100

cl_flask = np.zeros((nsims, lmax + 1))

for i in tqdm(range(nsims)):
    k = hp.ma(kappas[:,i])
    k.mask = ~mask

    cl_flask[i, :] = hp.anafast(k, lmax=lmax)

avg_cl_flask = cl_flask.mean(axis=0)
std_cl_flask = cl_flask.std(axis=0)

cov = np.load(out_dir + 'cov.npy')

u = np.load(out_dir + 'u.npy')
s = np.load(out_dir + 's.npy')

# plt.clf()
# plt.semilogy(s/s[0])
# plt.show()
# exit(0)

rcond = 0.03
good_vecs = s / s[0] > rcond

s = s[good_vecs]
u = u[:, good_vecs]

chain = np.load(out_dir + 'chain.npy')
logp = np.load(out_dir + 'logp.npy')

chain = (u @ chain.T).T

var = cov[0,0]
sigma = np.sqrt(var)
mu = -0.5 * var + np.log(shift)

chain = np.exp(chain + mu) - shift

rand_samps = np.random.choice(chain.shape[0], nsims, replace=False)

cl_samps = np.zeros_like(cl_flask)

ind = 0
for i in tqdm(rand_samps):
    m = np.zeros(hp.nside2npix(nside))
    m[mask] = chain[i, :]
    k = hp.ma(m)
    k.mask = ~mask

    cl_samps[ind, :] = hp.anafast(k, lmax=lmax)

    ind += 1

avg_cl_samps = cl_samps.mean(axis=0)
std_cl_samps = cl_samps.std(axis=0)

lrange = range(lmax + 1)

plt.plot(lrange, avg_cl_flask, label='{0} Masked Flask Realizations'.format(nsims))
plt.plot(lrange, avg_cl_samps, label='{0} Masked Sampler Realizations'.format(nsims))
plt.fill_between(lrange, avg_cl_flask - std_cl_flask, avg_cl_flask + std_cl_flask, alpha=0.5)
plt.fill_between(lrange, avg_cl_samps - std_cl_samps, avg_cl_samps + std_cl_samps, alpha=0.5)
plt.xlabel('$\\ell$')
plt.ylabel('$C_\\ell$')
plt.legend()
plt.title('$n_\\mathrm{side} = %d$' % nside)
plt.savefig(fig_dir + 'cl_comp.png', dpi=300, bbox_inches='tight')

plt.clf()
plt.plot(range(len(logp)), logp)
plt.xlabel('Sample #')
plt.ylabel('Log-Likelihood')
plt.tight_layout()
plt.savefig(fig_dir + 'logp', dpi=300, bbox_inches='tight')

plt.clf()
c = ChainConsumer()
c.add_chain(chain[:, :5])
c.plotter.plot(figsize='column')
plt.savefig(fig_dir + 'corner', dpi=300, bbox_inches='tight')
