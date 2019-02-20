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
    double *u;
    double *x_mu;
    double shift;
    double mu;
    double *inv_theory_cov;
    double *g1_obs;
    double *g2_obs;
    double *k2g1;
    double *k2g2;
    double *sn_var;
    int mask_npix;
    int buffered_npix;
} LikelihoodArgs;

SampleChain sample_map(double *x0, double *m, double *sigma_p, LikelihoodArgs args,
                       int num_samps, int num_steps, double epsilon, double T_scale);

Hamiltonian map_likelihood(double *y, void *args_ptr);

#endif //LIKELIHOOD_LIKELIHOOD_H
