//
// Created by pierfied on 6/2/18.
//

#include "likelihood.h"
#include "healpix_wrapper.h"
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
    double shift = args->shift;
    double *inv_cov = args->inv_cov;
    double *inv_resid_cov = args->inv_resid_cov;
    int num_params = args->num_params;
    int nside = args->nside;
    int lmax = args->lmax;
    int npix = 12 * nside * nside;

    int *y_inds = args->y_inds;

    double *y_obs = args->y_obs;
    double *kappa_obs = args->kappa_obs;

    double *kappa = malloc(sizeof(double) * npix);
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        kappa[i] = exp(y[i]) - shift;
    }

    double *grad = malloc(sizeof(double) * npix);

    double *gamma1_obs = args->gamma_1_obs;
    double *gamma2_obs = args->gamma_2_obs;
    double *shape_noise_1 = args->shape_noise_1;
    double *shape_noise_2 = args->shape_noise_2;

    Shears shears = conv2shear(nside, kappa, lmax);
    double *gamma1 = shears.gamma1;
    double *gamma2 = shears.gamma2;

    double normal_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < npix; i++) {
        int y1 = y_inds[i];

        // Loop over neighbors.
        double neighbor_contrib_theory = 0;
        double neighbor_contrib_resid = 0;
        for (int j = 0; j < npix; j++) {
            int y2 = y_inds[j];

            // Compute the neighbor contribution.
            int ic_ind = npix * i + j;
            neighbor_contrib_theory += inv_cov[ic_ind] * (y[j] - mu);

            if(y1 > 0 && y2 > 0){
                neighbor_contrib_resid += inv_resid_cov[ic_ind] * (kappa[j] - kappa_obs[j]);
            }
        }

        double shear_contrib = 0;
        if(y1 > 0){
            double delta_1 = gamma1[i] - gamma1_obs[i];
            double delta_2 = gamma2[i] - gamma2_obs[i];

            shear_contrib = delta_1 * delta_1 / shape_noise_1[i] + delta_2 * delta_2 / shape_noise_2[i];
        }

#pragma omp critical
        {
            // Compute the total contribution of this voxel to the normal.
//            normal_contrib += (y[i] - mu) * neighbor_contrib_theory + (kappa[i] - kappa_obs[i]) * neighbor_contrib_resid;
            normal_contrib += (y[i] - mu) * neighbor_contrib_theory + shear_contrib;
        }

        // Compute the gradient for the voxel.
        grad[i] = -neighbor_contrib_theory - (kappa[i] + shift) * neighbor_contrib_resid;
    }

//#pragma omp parallel for
//    for (int i = 0; i < npix; ++i) {
//        if(y_inds[i] > 0){
//            double delta_1 = gamma1[i] - gamma1_obs[i];
//            double delta_2 = gamma2[i] - gamma2_obs[i];
//            double shear_contrib = delta_1 * delta_1 / shape_noise_1[i] + delta_2 * delta_2 / shape_noise_2[i];
//
//#pragma omp critical
//            {
//                normal_contrib += shear_contrib;
//            }
//        }
//    }
    normal_contrib *= -0.5;

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib;
    likelihood.grad = grad;

    return likelihood;
}