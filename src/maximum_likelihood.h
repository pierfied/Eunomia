//
// Created by pierfied on 2/20/19.
//

#ifndef LIKELIHOOD_MAXIMUM_LIKELIHOOD_H
#define LIKELIHOOD_MAXIMUM_LIKELIHOOD_H


#ifdef __cplusplus
extern "C" {
#endif

void my_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df);

#ifdef __cplusplus
}
#endif


#endif //LIKELIHOOD_MAXIMUM_LIKELIHOOD_H
