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
    _fields_ = [('num_sing_vecs', ctypes.c_int),
                ('npix', ctypes.c_int),
                ('shift', ctypes.c_double),
                ('mu', ctypes.c_double),
                ('inv_s', ctypes.POINTER(ctypes.c_double)),
                ('u', ctypes.POINTER(ctypes.c_double)),
                ('k_obs', ctypes.POINTER(ctypes.c_double)),
                ('inv_noise_cov', ctypes.POINTER(ctypes.c_double))]


class MapSampler:
    def __init__(self, k_obs, shift, mu, s, u, inv_noise_cov):
        self.k_obs = k_obs
        self.shift = shift
        self.mu = mu
        self.s = s
        self.u = u
        self.inv_noise_cov = inv_noise_cov

    def sample(self, num_burn, num_burn_steps, burn_epsilon, num_samps, num_samp_steps, samp_epsilon):
        lib_path = os.path.join(os.path.dirname(__file__), '../lib/liblikelihood.so')
        sampler_lib = ctypes.cdll.LoadLibrary(lib_path)

        sampler = sampler_lib.sample_map
        sampler.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double), LikelihoodArgs,
                            ctypes.c_int, ctypes.c_int, ctypes.c_double,
                            ctypes.c_int, ctypes.c_int, ctypes.c_double]
        sampler.restype = SampleChain

        # jacobian = np.diag(1/(1 + self.kappa_obs))
        # self.cov = np.matmul(np.matmul(jacobian, self.cov), jacobian)

        # var = self.cov[0,0]
        # sigma = np.sqrt(var)
        # mu = -0.5 * var + np.log(self.shift)
        #
        # epsilon *= sigma

        num_sing_vecs = len(self.s)

        k_obs = np.ascontiguousarray(self.k_obs, dtype=np.double)
        inv_s = np.ascontiguousarray(1 / self.s.ravel(), dtype=np.double)
        u = np.ascontiguousarray(self.u.ravel(), dtype=np.double)
        inv_noise_cov = np.ascontiguousarray(self.inv_noise_cov.ravel(), dtype=np.double)

        # print(s)
        # print(s.shape)
        # print(u.shape)
        # exit(0)

        args = LikelihoodArgs()
        args.num_sing_vecs = self.u.shape[1]
        args.npix = self.u.shape[0]
        args.shift = self.shift
        args.mu = self.mu
        args.inv_s = inv_s.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.k_obs = k_obs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.inv_noise_cov = inv_noise_cov.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # y0 = np.ascontiguousarray(mu + np.random.standard_normal(num_y_params) * sigma)
        # y0_p = y0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        x0 = np.ascontiguousarray(np.random.standard_normal(num_sing_vecs) * np.sqrt(self.s))
        x0_p = x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # m = np.ascontiguousarray(np.ones(num_sing_vecs, dtype=np.double))
        m = np.ascontiguousarray(1 / self.s)
        m_p = m.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        sigmap = np.ascontiguousarray(1 / np.sqrt(self.s))
        sigmap_p = sigmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        print('Starting Sampling')

        results = sampler(x0_p, m_p, sigmap_p, args, num_burn, num_burn_steps, burn_epsilon,
                          num_samps, num_samp_steps, samp_epsilon)

        print('Sampling Finished')

        print(results.accept_rate)

        chain = np.array([[results.samples[i][j] for j in range(num_sing_vecs)]
                          for i in range(num_samps)])

        likelihoods = np.array([results.log_likelihoods[i]
                                for i in range(num_samps)])

        return chain, likelihoods
