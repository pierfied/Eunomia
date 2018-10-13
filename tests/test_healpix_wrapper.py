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

double_ptr = ctypes.POINTER(ctypes.c_double)

class Shears(ctypes.Structure):
    _fields_ = [('gamma1', double_ptr),
                ('gamma2', double_ptr)]

conv2shear = sampler_lib.conv2shear
conv2shear.argtypes = [ctypes.c_int, ctypes.c_int, double_ptr, ctypes.c_int]
conv2shear.restype = Shears

shear2conv = sampler_lib.shear2conv
shear2conv.argtypes = [ctypes.c_int, ctypes.c_int, Shears, ctypes.c_int]
shear2conv.restype = double_ptr

map = np.load('./test_data/kappas_64_64.npy')[:,0].ravel()
in_nside = 64
out_nside = 16
in_npix = hp.nside2npix(in_nside)
out_npix = hp.nside2npix(out_nside)
lmax = 64

print('Here')
shears = conv2shear(in_nside, in_nside, map.ctypes.data_as(double_ptr), lmax)
print('here2')
kappa_ptr = shear2conv(in_nside, out_nside, shears, lmax)

kappa = np.array([kappa_ptr[i] for i in range(out_npix)])
py_kappa = pyshear2conv(pyconv2shear(map))[0]

print(map)
print(kappa)
print(py_kappa)

# map2alm2map = sampler_lib.map2alm2map
# map2alm2map.argtypes = [ctypes.c_int, double_ptr, ctypes.c_int]
# map2alm2map.restype = ctypes.POINTER(ctypes.c_double)
#
# map_recov_ptr = map2alm2map(npix, map.ctypes.data_as(double_ptr), lmax)
# map_recov = np.array([map_recov_ptr[i] for i in range(out_npix)])
# py_map_recov = hp.alm2map(hp.map2alm(map, lmax), nside, lmax)
#
# print('\n\n\n\n\n\n\n\n')
#
# print(map)
# print(map_recov)
# print(py_map_recov)

import matplotlib.pyplot as plt
mapu = hp.ud_grade(map, out_nside)
plt.hist((kappa - mapu)/mapu.std(), 20)
plt.show()