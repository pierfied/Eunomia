import numpy as np
import healpy as hp
import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), '../lib/liblikelihood.so')
sampler_lib = ctypes.cdll.LoadLibrary(lib_path)

test_alm = sampler_lib.test_alm
test_alm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

nside = 16
lmax = nside
npix = hp.nside2npix(nside)
np.random.seed(0)
map = np.random.standard_normal(npix)
test_alm(lmax, npix, map.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

l, m = hp.Alm.getlm(lmax)
alm = hp.map2alm(map, lmax)

print(alm[(l == 16) & (m == 0)])
print(alm[(l == 1) & (m == 1)])

def pyconv2shear(kappa):
    nside = hp.npix2nside(len(kappa))

    k = kappa
    ke = kappa
    kb = np.zeros(len(kappa))

    keb = [k, ke, kb]

    klm, kelm, kblm = hp.map2alm(keb, pol=False, lmax=lmax)

    l, _ = hp.Alm.getlm(lmax)

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

def pyshear2conv(gqu):
    nside = hp.npix2nside(gqu.shape[1])

    glm, gelm, gblm = hp.map2alm(gqu, lmax=lmax)

    l, _ = hp.Alm.getlm(lmax)

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

class Shears(ctypes.Structure):
    _fields_ = [('gamma1', ctypes.POINTER(ctypes.c_double)),
                ('gamma2', ctypes.POINTER(ctypes.c_double))]

conv2shear = sampler_lib.conv2shear
conv2shear.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
conv2shear.restype = Shears

s = conv2shear(npix, map.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lmax)

g1 = np.array([s.gamma1[i] for i in range(npix)])
g2 = np.array([s.gamma2[i] for i in range(npix)])

print(g1)
print(g2)

g1 = pyconv2shear(map)[1,:]
g2 = pyconv2shear(map)[2,:]

print(g1)
print(g2)