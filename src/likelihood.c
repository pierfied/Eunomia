//
// Created by pierfied on 6/2/18.
//

#include "likelihood.h"
#include "../include/hmc.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

SampleChain sample_map(double *y0, double *m, LikelihoodArgs args,
                       int num_samps, int num_steps, int num_burn,
                       double epsilon) {
    HMCArgs hmc_args;
    hmc_args.log_likelihood = map_likelihood;
    hmc_args.likelihood_args = &args;
    hmc_args.num_samples = num_samps;
    hmc_args.num_params = args.num_params;
    hmc_args.num_steps = num_steps;
    hmc_args.num_burn = num_burn;
    hmc_args.epsilon = epsilon;
    hmc_args.x0 = y0;
    hmc_args.m = m;

    SampleChain chain = hmc(hmc_args);

    return chain;
}

Hamiltonian map_likelihood(double *y, void *args_ptr) {
    LikelihoodArgs *args = (LikelihoodArgs *) args_ptr;

    double mu = args->mu;
    double *inv_cov = args->inv_cov;
    double *g1_obs = args->g1_obs;
    double *g2_obs = args->g2_obs;
    double *k2g1 = args->k2g1;
    double *k2g2 = args->k2g2;
    double sn_var = args->sn_var;
    double shift = args->shift;

    int *y_inds = args->y_inds;

    int num_params = args->num_params;
    double *grad = malloc(sizeof(double) * num_params);

    double exp_y[num_params];
    double g1[num_params];
    double g2[num_params];
#pragma omp parallel for
    for (int i = 0; i < num_params; ++i) {
        exp_y[i] = exp(y[i]);
        g1[i] = 0;
        g2[i] = 0;
    }

    for (int i = 0; i < num_params; ++i) {
        double kappa_i = exp_y[i] - shift;

#pragma omp parallel for
        for (int j = 0; j < num_params; ++j) {
            int mat_ind = i * num_params + j;
            g1[j] += k2g1[mat_ind] * kappa_i;
            g2[j] += k2g2[mat_ind] * kappa_i;
        }
    }

//    printf("Num Params: %d\n", num_params);

    double normal_contrib = 0;
    double shear_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < num_params; i++) {
        // Loop over neighbors.
        double neighbor_contrib = 0;
        double g1_grad_term = 0;
        double g2_grad_term = 0;
        for (int j = 0; j < num_params; j++) {
            // Compute the neighbor contribution.
            int ic_ind = num_params * i + j;
            neighbor_contrib += inv_cov[ic_ind] * (y[j] - mu);

            ic_ind = num_params * i + j;
            g1_grad_term += (g1_obs[j] - g1[j]) * k2g1[ic_ind] * exp_y[i];
            g2_grad_term += (g2_obs[j] - g2[j]) * k2g2[ic_ind] * exp_y[i];
        }

        double delta_gamma_1 = (g1_obs[i] - g1[i]);
        double delta_gamma_2 = (g2_obs[i] - g2[i]);

#pragma omp critical
        {
            // Compute the total contribution of this voxel to the normal.
            normal_contrib += (y[i] - mu) * neighbor_contrib;

            shear_contrib += delta_gamma_1 * delta_gamma_1 + delta_gamma_2 * delta_gamma_2;
        }

        // Compute the gradient for the voxel.
        grad[i] = -neighbor_contrib + (g1_grad_term + g2_grad_term) / sn_var;
    }
    normal_contrib *= -0.5;
    shear_contrib *= -0.5 / sn_var;

//    printf("Normal Contrib: %f\n", normal_contrib);

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib + shear_contrib;
    likelihood.grad = grad;

    return likelihood;
}