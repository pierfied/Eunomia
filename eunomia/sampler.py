import numpy as np
import ctypes
import os

class SampleChain(ctypes.Structure):
    _fields_ = [('num_samples', ctypes.c_int),
                ('num_params', ctypes.c_int),
                ('accept_rate', ctypes.c_double),
                ('samples', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ('log_likelihoods', ctypes.POINTER(ctypes.c_double))]

class LikelihoodArgs(ctypes.Structure):
    _fields_ = [('num_params', ctypes.c_int),
                ('y_inds', ctypes.POINTER(ctypes.c_int)),
                ('mu', ctypes.c_double),
                ('shift', ctypes.c_double),
                ('inv_cov', ctypes.POINTER(ctypes.c_double)),
                ('inv_resid_cov', ctypes.POINTER(ctypes.c_double)),
                ('shape_noise_1', ctypes.POINTER(ctypes.c_double)),
                ('shape_noise_2', ctypes.POINTER(ctypes.c_double)),
                ('y_obs', ctypes.POINTER(ctypes.c_double)),
                ('kappa_obs', ctypes.POINTER(ctypes.c_double)),
                ('gamma1_obs', ctypes.POINTER(ctypes.c_double)),
                ('gamma2_obs', ctypes.POINTER(ctypes.c_double)),
                ('kappa_nside', ctypes.c_int),
                ('gamma_nside', ctypes.c_int),
                ('lmax', ctypes.c_int)]

class MapSampler:
    def __init__(self, y_obs, cov, inv_cov, shift, resid_cov=None):
        self.y_obs = y_obs
        self.cov = cov
        self.inv_cov = inv_cov
        self.shift = shift
        self.resid_cov = resid_cov

    def sample(self, num_samps, num_steps, num_burn, epsilon):
        lib_path = os.path.join(os.path.dirname(__file__), '../lib/liblikelihood.so')
        sampler_lib = ctypes.cdll.LoadLibrary(lib_path)

        sampler = sampler_lib.sample_map
        sampler.argtypes = [ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            LikelihoodArgs, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_double]
        sampler.restype = SampleChain

        var = self.cov[0,0]
        sigma = np.sqrt(var)
        mu = -0.5 * var + np.log(self.shift)

        epsilon *= sigma

        # inv_cov = np.linalg.inv(self.cov)
        inv_cov = self.inv_cov

        if self.resid_cov is not None:
            inv_resid_cov = np.linalg.inv(self.resid_cov)
        else:
            inv_resid_cov = np.zeros(self.cov.shape, dtype=np.double)

        num_params = self.cov.shape[0]
        y_inds = np.arange(num_params, dtype=np.int32)

        args = LikelihoodArgs()
        args.num_params = num_params
        args.y_inds = y_inds.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        args.mu = mu
        args.inv_cov = inv_cov.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.inv_resid_cov = inv_resid_cov.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.y_obs = self.y_obs.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.shift = self.shift

        y0 = np.random.normal(mu, sigma, num_params)
        # y0 = self.y_obs.ravel()
        # y0 = np.random.multivariate_normal(np.array([mu for i in range(num_params)]), self.cov)
        y0_p = y0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        m = np.ones(num_params, dtype=np.double)
        m_p = m.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        print('Starting Sampling')

        results = sampler(y0_p, m_p, args, num_samps, num_steps, num_burn, epsilon)

        print('Sampling Finished')

        print(results.accept_rate)

        chain = np.array([[results.samples[i][j] for j in range(num_params)]
                          for i in range(num_samps)])

        likelihoods = np.array([results.log_likelihoods[i]
                                for i in range(num_samps)])

        return chain, likelihoods