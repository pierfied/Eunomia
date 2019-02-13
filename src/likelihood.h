//
// Created by pierfied on 6/2/18.
//

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include "../include/hmc.h"

#ifndef LIKELIHOOD_LIKELIHOOD_H
#define LIKELIHOOD_LIKELIHOOD_H

typedef struct {
    int num_sing_vecs;
    double *u_joint;
    double *mu_joint;
    int npix;
    double shift;
    double mu;
    double *inv_theory_cov;
    double *k_obs;
    double *inv_noise_cov;
} LikelihoodArgs;

SampleChain sample_map(double *x0, double *m, double *sigma_p, LikelihoodArgs args,
                       int num_burn, int num_burn_steps, double burn_epsilon,
                       int num_samps, int num_samp_steps, double samp_epsilon);

Hamiltonian map_likelihood(double *y, void *args_ptr);

#endif //LIKELIHOOD_LIKELIHOOD_H
