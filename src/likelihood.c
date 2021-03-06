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

    int *y_inds = args->y_inds;

    int num_params = args->num_params;
    double *grad = malloc(sizeof(double) * num_params);

    double normal_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < num_params; i++) {
        // Check that this is a high occupancy voxel.
        if (y_inds[i] < 0) continue;
        int y1 = y_inds[i];

        // Loop over neighbors.
        double neighbor_contrib = 0;
        for (int j = 0; j < num_params; j++) {
            if (y_inds[j] < 0) continue;
            int y2 = y_inds[j];

            // Compute the neighbor contribution.
            int ic_ind = num_params * i + j;
            neighbor_contrib += inv_cov[ic_ind] * (y[y2] - mu);
        }

#pragma omp critical
        {
            // Compute the total contribution of this voxel to the normal.
            normal_contrib += (y[y1] - mu) * neighbor_contrib;
        }

        // Compute the gradient for the voxel.
        grad[y1] = -neighbor_contrib;
    }
    normal_contrib *= -0.5;

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib;
    likelihood.grad = grad;

    return likelihood;
}