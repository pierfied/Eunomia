//
// Created by pierfied on 6/2/18.
//

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include "../include/hmc.h"

#ifndef LIKELIHOOD_LIKELIHOOD_H
#define LIKELIHOOD_LIKELIHOOD_H

typedef struct {
    int num_y_params;
    int *y_inds;
    double shift;
    double mu;
    double *inv_s;
    double *v;
    double *g1_obs;
    double *g2_obs;
    double *k2g1;
    double *k2g2;
    double sn_var;
    int num_sing_vals;
    gsl_matrix_view *v_mat;
} LikelihoodArgs;

SampleChain sample_map(double *y0, double *m, LikelihoodArgs args,
                       int num_samps, int num_steps, int num_burn,
                       double epsilon);

Hamiltonian map_likelihood(double *y, void *args_ptr);

#endif //LIKELIHOOD_LIKELIHOOD_H
