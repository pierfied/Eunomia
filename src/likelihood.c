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

    int num_sing_vecs = args->num_sing_vecs;
    double *u_joint = args->u_joint;
    double *mu_joint = args->mu_joint;
    int npix = args->npix;
    double mu = args->mu;
    double shift = args->shift;
    double *inv_theory_cov = args->inv_theory_cov;
    double *k_obs = args->k_obs;
    double *inv_noise_cov = args->inv_noise_cov;

//    for (int i = 0; i < npix; ++i) {
//        printf("%f\n", inv_theory_cov[i]);
//    }
//    exit(0);

    double exp_y[npix];
    double delta_y[npix];
    double kappa[npix];
    double delta_kappa[npix];
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        double y_i = 0;

        for (int j = 0; j < num_sing_vecs; ++j) {
            y_i += u_joint[i * num_sing_vecs + j] * params[j];
        }

        y_i += mu_joint[i];

        delta_y[i] = y_i - mu;
        exp_y[i] = exp(y_i);
        kappa[i] = exp_y[i] - shift;
        delta_kappa[i] = (kappa[i] - k_obs[i]);
    }

    double *grad = malloc(sizeof(double) * num_sing_vecs);

    double df_dy[npix];
    double normal_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        double normal_contrib_i = 0;

        for (int j = 0; j < npix; ++j) {
            normal_contrib_i += inv_theory_cov[i * npix + j] * delta_y[j];
        }

        df_dy[i] = -normal_contrib_i;

        normal_contrib_i *= delta_y[i];

#pragma omp atomic
        normal_contrib += normal_contrib_i;
    }
    normal_contrib *= -0.5;

    double df_dx[num_sing_vecs];
#pragma omp parallel for
    for (int i = 0; i < num_sing_vecs; ++i) {
        double df_dx_i = 0;

        for (int j = 0; j < npix; ++j) {
            df_dx_i += df_dy[j] * u_joint[j * num_sing_vecs + i];
        }

        df_dx[i] = df_dx_i;
    }

    double dg_dk[npix];
    double noise_contrib = 0;
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        double noise_contrib_i = 0;

        for (int j = 0; j < npix; ++j) {
            noise_contrib_i += inv_noise_cov[i * npix + j] * delta_kappa[j];
        }

        dg_dk[i] = -noise_contrib_i;

        noise_contrib_i *= delta_kappa[i];

#pragma omp atomic
        noise_contrib += noise_contrib_i;
    }
    noise_contrib *= -0.5;

    double dg_dy[npix];
#pragma omp parallel for
    for (int i = 0; i < npix; ++i) {
        dg_dy[i] = dg_dk[i] * exp_y[i];
    }

    double dg_dx[num_sing_vecs];
#pragma omp parallel for
    for (int i = 0; i < num_sing_vecs; ++i) {
        double dg_dx_i = 0;

        for (int j = 0; j < npix; ++j) {
            dg_dx_i += dg_dy[j] * u_joint[j * num_sing_vecs + i];
        }

        dg_dx[i] = dg_dx_i;
    }

#pragma omp parallel for
    for (int i = 0; i < num_sing_vecs; ++i) {
        grad[i] = df_dx[i] + dg_dx[i];
    }

//    printf("%f\n", normal_contrib);
//    printf("%f\n", noise_contrib);
//    exit(0);

    Hamiltonian likelihood;
    likelihood.log_likelihood = normal_contrib + noise_contrib;
    likelihood.grad = grad;

    return likelihood;
}