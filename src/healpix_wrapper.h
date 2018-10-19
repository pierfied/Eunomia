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

    Shears conv2shear(int in_nside, int out_nside, double *raw_map, int lmax);
    double *shear2conv(int in_nside, int out_nside, Shears shears, int lmax);

#ifdef __cplusplus
} //end extern "C"
#endif

#endif //LIKELIHOOD_HEALPIX_WRAPPER_H
