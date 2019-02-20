//
// Created by pierfied on 2/20/19.
//

#include <gsl/gsl_multimin.h>
#include "maximum_likelihood.h"
#include "likelihood.h"

void my_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df) {
    Hamiltonian ml = map_likelihood(x->data, params);

    *f = -ml.log_likelihood;

#pragma omp parallel for
    for (int i = 0; i < x->size; ++i) {
        gsl_vector_set(df, i, -ml.grad[i]);
    }
}

double *minimize(double *x0, LikelihoodArgs *args) {
    gsl_multimin_function_fdf my_func;

    my_func.n = args->num_sing_vecs;
    my_func.fdf = &my_fdf;
    my_func.params = (void *) args;

    gsl_vector_view x = gsl_vector_view_array(x0, args->num_sing_vecs);

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc(T, args->num_sing_vecs);

    gsl_multimin_fdfminimizer_set(s, &my_func, &x.vector, 0.01, 1e-4);

    int iter = 0;
    int status;
    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);

        if (status)
            break;

        status = gsl_multimin_test_gradient(s->gradient, 1e-3);

        if (status == GSL_SUCCESS)
            printf("Minimum found at:\n");

        printf("%5d %.5f %.5f %10.5f\n", iter,
               gsl_vector_get(s->x, 0),
               gsl_vector_get(s->x, 1),
               s->f);

    } while (status == GSL_CONTINUE && iter < 100);

    gsl_multimin_fdfminimizer_free(s);

    return x.vector.data;
}