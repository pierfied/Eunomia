import numpy as np
import healpy as hp
from tqdm import tqdm

def conv2shear(k, lmax):
    nside = hp.npix2nside(len(k))

    kelm = hp.map2alm(k, lmax=lmax)

    l, m = hp.Alm.getlm(lmax)

    kelm[l == 0] = 0

    gelm = - np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kelm
    gblm = np.zeros_like(gelm)
    glm = gelm + 1j * gblm

    _, g1, g2 = hp.alm2map([glm, gelm, gblm], nside, lmax=lmax, verbose=False)

    return g1, g2


def shear2conv(g1, g2, lmax):
    nside = hp.npix2nside(len(g1))

    g = g1 + 1j * g2

    _, gelm, _ = hp.map2alm([g, g1, g2], lmax=lmax)

    l, m = hp.Alm.getlm(lmax)

    kelm = - np.sqrt(l * (l + 1) / ((l + 2) * (l - 1))) * gelm

    kelm[l == 0] = 0
    kelm[l == 1] = 0

    k = hp.alm2map(kelm, nside, lmax=lmax, verbose=False)

    return k


def compute_full_conv2shear_mats(nside, lmax, mask):
    npix = hp.nside2npix(nside)

    A1 = np.zeros((mask.sum(), mask.sum()))
    A2 = np.zeros_like(A1)

    mask_inds = np.arange(npix)[mask]

    mat_ind = 0
    for i in tqdm(mask_inds):
        k = np.zeros(npix)
        k[i] = 1

        g1, g2 = conv2shear(k, lmax)

        A1[:, mat_ind] = g1[mask_inds]
        A2[:, mat_ind] = g2[mask_inds]

        mat_ind += 1

    return A1, A2
