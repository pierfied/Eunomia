//
// Created by pierfied on 6/2/18.
//

#include "likelihood.h"
#include "../include/hmc.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

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
    for (int i = 0; i < num_params; ++i) {
        exp_y[i] = exp(y[i]);
    }

    double normal_contrib = 0;
    double shear_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < num_params; i++) {
        // Check that this is a high occupancy voxel.
        if (y_inds[i] < 0) continue;
        int y1 = y_inds[i];

        // Loop over neighbors.
        double neighbor_contrib = 0;
        double g1 = 0;
        double g2 = 0;
        for (int j = 0; j < num_params; j++) {
            if (y_inds[j] < 0) continue;
            int y2 = y_inds[j];

            // Compute the neighbor contribution.
            int ic_ind = num_params * i + j;
            neighbor_contrib += inv_cov[ic_ind] * (y[y2] - mu);
            g1 += k2g1[ic_ind] * (exp_y[y2] - shift);
            g2 += k2g2[ic_ind] * (exp_y[y2] - shift);
        }

        double delta_gamma_1 = (g1_obs[y1] - g1);
        double delta_gamma_2 = (g2_obs[y1] - g2);

#pragma omp critical
        {
            // Compute the total contribution of this voxel to the normal.
            normal_contrib += (y[y1] - mu) * neighbor_contrib;

            shear_contrib += delta_gamma_1 * delta_gamma_1 + delta_gamma_2 * delta_gamma_2;
        }

        // Compute the gradient for the voxel.
        int ind = num_params * i + i;
        grad[y1] = -neighbor_contrib +
                   (delta_gamma_1 * k2g1[ind] * exp_y[y1] + delta_gamma_2 * k2g2[ind] * exp_y[y1]) / sn_var;
    }
    normal_contrib *= -0.5;
    shear_contrib *= -0.5 / sn_var;

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib;
    likelihood.grad = grad;

    return likelihood;
}