import numpy as np
import healpy as hp


def cov_sep_theta_from_cl(theta, cl):
    # Compute the Legendre polynomial coefficients.
    lmax = len(cl) - 1
    legendre_coeffs = (2 * np.arange(lmax + 1) + 1) / (4 * np.pi) * cl

    # Calculate the covariance at the given separation.
    return np.polynomial.legendre.legval(np.cos(theta), legendre_coeffs)


def full_cov_from_cl(cl, nside, indices=None):
    # If indices not passed construct full-sky indices.
    if indices is None:
        indices = np.arange(hp.nside2npix(nside))

    # Get the number of pixels.
    npix = len(indices)

    # Compute the angular separation of each pair of pixels.
    theta, phi = hp.pixelfunc.pix2ang(nside, indices)
    ang_coords = np.stack([theta, phi])
    ang_sep = np.zeros([npix, npix])
    for i in range(npix):
        ang_sep[i, :] = hp.rotator.angdist(ang_coords[:, i], ang_coords)

    # Compute the full covariance matrix.
    cov = cov_sep_theta_from_cl(ang_sep, cl)

    return cov, ang_sep

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
        ang_sep[i, neighbors] = hp.rotator.angdist(ang_coords[:, i], ang_coords[:, neighbors])
        cov[i, neighbors] = cov_sep_theta_from_cl(ang_sep[i, neighbors], cl)

    return cov, ang_sep