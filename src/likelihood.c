//
// Created by pierfied on 6/2/18.
//

#include "likelihood.h"
#include "../include/hmc.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <gsl/gsl_linalg.h>

SampleChain sample_map(double *x0, double *m, double *sigma_p, LikelihoodArgs args,
                       int num_burn, int num_burn_steps, double burn_epsilon,
                       int num_samps, int num_samp_steps, double samp_epsilon) {
    HMCArgs hmc_args;
    hmc_args.log_likelihood = map_likelihood;
    hmc_args.likelihood_args = &args;
    hmc_args.num_params = args.num_sing_vecs;
    hmc_args.num_burn = num_burn;
    hmc_args.num_burn_steps = num_burn_steps;
    hmc_args.burn_epsilon = burn_epsilon;
    hmc_args.num_samples = num_samps;
    hmc_args.num_samp_steps = num_samp_steps;
    hmc_args.samp_epsilon = samp_epsilon;
    hmc_args.x0 = x0;
    hmc_args.m = m;
    hmc_args.sigma_p = sigma_p;

    SampleChain chain = hmc(hmc_args);

    return chain;
}

Hamiltonian map_likelihood(double *params, void *args_ptr) {
    LikelihoodArgs *args = (LikelihoodArgs *) args_ptr;

    double mu = args->mu;
    double *inv_s = args->inv_s;
    double *u = args->u;
    double *g1_obs = args->g1_obs;
    double *g2_obs = args->g2_obs;
    double *k2g1 = args->k2g1;
    double *k2g2 = args->k2g2;
    double sn_var = args->sn_var;
    double shift = args->shift;
    int num_sing_vecs = args->num_sing_vecs;
    int mask_npix = args->mask_npix;
    int buffered_npix = args->buffered_npix;

    double exp_y[buffered_npix];
#pragma omp parallel for
    for (int i = 0; i < buffered_npix; ++i) {
        double y_i = 0;

        for (int j = 0; j < num_sing_vecs; ++j) {
            y_i += u[i * num_sing_vecs + j] * params[j];
        }

        exp_y[i] = exp(y_i + mu);
    }

    double g1[mask_npix];
    double g2[mask_npix];
    double delta_g1[mask_npix];
    double delta_g2[mask_npix];
#pragma omp parallel for
    for (int i = 0; i < mask_npix; ++i) {
        double g1_i = 0;
        double g2_i = 0;

        for (int j = 0; j < buffered_npix; ++j) {
            double kappa_j = exp_y[j] - shift;

            int ind = i * buffered_npix + j;

            g1_i += k2g1[ind] * kappa_j;
            g2_i += k2g2[ind] * kappa_j;
        }

        g1[i] = g1_i;
        g2[i] = g2_i;

        delta_g1[i] = g1_i - g1_obs[i];
        delta_g2[i] = g2_i - g2_obs[i];
    }

    double *grad = malloc(sizeof(double) * num_sing_vecs);

    double normal_contrib = 0;
    for (int i = 0; i < num_sing_vecs; ++i) {
        normal_contrib += params[i] * inv_s[i] * params[i];
    }
    normal_contrib *= -0.5;

    double shear_contrib = 0;
    for (int i = 0; i < mask_npix; ++i) {
        shear_contrib += delta_g1[i] * delta_g1[i] + delta_g2[i] * delta_g2[i];
    }
    shear_contrib /= -2 * sn_var;


    double df1_dg1[mask_npix];
    double df2_dg2[mask_npix];
#pragma omp parallel for
    for (int i = 0; i < mask_npix; ++i) {
        df1_dg1[i] = -delta_g1[i] / sn_var;
        df2_dg2[i] = -delta_g2[i] / sn_var;
    }

    double df1_dy[buffered_npix];
    double df2_dy[buffered_npix];
#pragma omp parallel for
    for (int i = 0; i < buffered_npix; ++i) {
        df1_dy[i] = 0;
        df2_dy[i] = 0;

        for (int j = 0; j < mask_npix; ++j) {
            int ind = j * buffered_npix + i;

            df1_dy[i] += df1_dg1[j] * k2g1[ind];
            df2_dy[i] += df2_dg2[j] * k2g2[ind];
        }

        df1_dy[i] *= exp_y[i];
        df2_dy[i] *= exp_y[i];
    }

    double df1_dx[num_sing_vecs];
    double df2_dx[num_sing_vecs];
#pragma omp parallel for
    for (int i = 0; i < num_sing_vecs; ++i) {
        df1_dx[i] = 0;
        df2_dx[i] = 0;

        for (int j = 0; j < buffered_npix; ++j) {
            int ind = j * num_sing_vecs + i;

            df1_dx[i] += df1_dy[j] * u[ind];
            df2_dx[i] += df2_dy[j] * u[ind];
        }
    }

#pragma omp parallel for
    for (int i = 0; i < num_sing_vecs; ++i) {
        grad[i] = -inv_s[i] * params[i] + df1_dx[i] + df2_dx[i];
    }

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib + shear_contrib;
    likelihood.grad = grad;

    return likelihood;
}