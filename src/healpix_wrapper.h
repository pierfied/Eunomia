//
// Created by pierfied on 10/2/18.
//

#ifndef LIKELIHOOD_HEALPIX_WRAPPER_H
#define LIKELIHOOD_HEALPIX_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct{
        double *gamma1;
        double *gamma2;
    } Shears;

    void test_alm(int lmax, int npix, double *raw_map);
    Shears conv2shear(int kappa_nside, int gamma_nside, double *raw_map, int lmax);
    double *shear2conv(int gamma_nside, int kappa_nside, Shears shears, int lmax);
    double *map2alm2map(int npix, double *raw_map, int lmax);

#ifdef __cplusplus
} //end extern "C"
#endif

#endif //LIKELIHOOD_HEALPIX_WRAPPER_H
