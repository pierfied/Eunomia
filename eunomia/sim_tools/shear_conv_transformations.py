import numpy as np
import healpy as hp


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


def compute_full_conv2shear_mats(nside):
    npix = hp.nside2npix(nside)

    A1 = np.zeros((npix, npix))
    A2 = np.zeros_like(A1)

    for i in range(npix):
        k = np.zeros(npix)
        k[i] = 1

        g1, g2 = conv2shear(k, 64)

        A1[:, i] = g1
        A2[:, i] = g2

    return A1, A2
