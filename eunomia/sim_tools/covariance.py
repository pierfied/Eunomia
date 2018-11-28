import numpy as np
import healpy as hp
import scipy.interpolate as interp
import scipy.sparse as sp
from tqdm import tqdm


def cov_sep_theta_from_cl(theta, cl):
    # Compute the Legendre polynomial coefficients.
    lmax = len(cl) - 1
    legendre_coeffs = (2 * np.arange(lmax + 1) + 1) / (4 * np.pi) * cl

    # Calculate the covariance at the given separation.
    return np.polynomial.legendre.legval(np.cos(theta), legendre_coeffs)


def full_cov_from_cl(cl, nside, max_sep, indices=None):
    # If indices not passed construct full-sky indices.
    if indices is None:
        indices = np.arange(hp.nside2npix(nside))

    thetas = np.linspace(0, np.pi, 10000)
    cov_seps = cov_sep_theta_from_cl(thetas, cl)

    s = interp.InterpolatedUnivariateSpline(thetas, cov_seps)

    # Get the number of pixels.
    npix = len(indices)

    cov = sp.lil_matrix((npix, npix))

    print("Computing Sparse Theory Covariance")

    for i in tqdm(range(npix)):
        pix_vec = hp.pix2vec(nside, i)
        neighbor_pixs = hp.query_disc(nside, pix_vec, max_sep)
        neighbor_vecs = hp.pix2vec(nside, neighbor_pixs)
        neighbor_seps =  hp.rotator.angdist(pix_vec, neighbor_vecs)

        neighbor_covs = cov_sep_theta_from_cl(neighbor_seps, cl)

        cov[i,neighbor_pixs] = neighbor_covs

    return sp.csr_matrix(cov)

def full_neighbor_cov_from_cl(cl, nside, indices=None):
    # If indices not passed construct full-sky indices.
    if indices is None:
        indices = np.arange(hp.nside2npix(nside))

    # Get the number of pixels.
    npix = len(indices)

    # Compute the angular separation of each pair of pixels.
    theta, phi = hp.pixelfunc.pix2ang(nside, indices)
    ang_coords = np.stack([theta, phi])
    ang_sep = np.zeros([npix, npix])
    cov = np.zeros([npix, npix])
    for i in range(npix):
        neighbors = np.concatenate((hp.pixelfunc.get_all_neighbours(nside, i), [i]))
        neighbors = np.intersect1d(neighbors, indices)
        ang_sep[i, neighbors] = hp.rotator.angdist(ang_coords[:, i], ang_coords[:, neighbors])
        cov[i, neighbors] = cov_sep_theta_from_cl(ang_sep[i, neighbors], cl)

    return cov, ang_sep