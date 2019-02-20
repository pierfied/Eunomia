//
// Created by pierfied on 2/20/19.
//

#include "likelihood.h"

#ifndef LIKELIHOOD_MAXIMUM_LIKELIHOOD_H
#define LIKELIHOOD_MAXIMUM_LIKELIHOOD_H


double my_f(const gsl_vector *x, void *params);
void my_df(const gsl_vector *x, void *params, gsl_vector *df);
void my_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df);

//double *maximizer(double *x0, double *s, LikelihoodArgs args);
double *maximizer(double *x0, LikelihoodArgs args);


#endif //LIKELIHOOD_MAXIMUM_LIKELIHOOD_H
