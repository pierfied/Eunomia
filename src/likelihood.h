//
// Created by pierfied on 6/2/18.
//

#include "../include/hmc.h"

#ifndef LIKELIHOOD_LIKELIHOOD_H
#define LIKELIHOOD_LIKELIHOOD_H

typedef struct {
    int num_params;
    int *y_inds;
    double shift;
    double mu;
    double *inv_cov;
    double *g1_obs;
    double *g2_obs;
    double *k2g1;
    double *k2g2;
} LikelihoodArgs;

SampleChain sample_map(double *y0, double *m, LikelihoodArgs args,
                       int num_samps, int num_steps, int num_burn,
                       double epsilon);

Hamiltonian map_likelihood(double *y, void *args_ptr);

#endif //LIKELIHOOD_LIKELIHOOD_H
