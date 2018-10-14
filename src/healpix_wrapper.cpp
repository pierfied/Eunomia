//
// Created by pierfied on 10/2/18.
//

#include "healpix_wrapper.h"
#include <healpix_map.h>
#include <alm.h>
#include <alm_healpix_tools.h>
#include <datatypes.h>
#include <iostream>
#include <arr.h>

using namespace std;

extern "C"{
    Shears conv2shear(int in_nside, int out_nside, double *raw_map, int lmax){
        int in_npix = 12 * in_nside * in_nside;
        int out_npix = 12 * out_nside * out_nside;

        arr<double> map_arr(raw_map, in_npix);
        Healpix_Map<double> kappa_map(map_arr, RING);

        Alm<xcomplex<double>> kappa_alms(lmax, lmax);
        arr<double> weights(2 * in_nside, 1);

        map2alm_iter(kappa_map, kappa_alms, 0, weights);

        Alm<xcomplex<double>> gamma_Elms(lmax, lmax);
        Alm<xcomplex<double>> gamma_Blms(lmax, lmax);

        xcomplex<double> *klm;
        xcomplex<double> *glm;
        glm = gamma_Elms.mstart(0);
        *glm = 0;
        for (int l = 1; l <= lmax; ++l) {
            for (int m = 0; m <= l; ++m) {
                klm = kappa_alms.mstart(m) + l;
                glm = gamma_Elms.mstart(m) + l;

                *glm = -sqrt((double)(l + 2) * (l - 1) / (l * (l + 1))) * (*klm);
            }
        }

        const nside_dummy dummy;
        Healpix_Map<double> gamma_in_res_map_T(in_nside, RING, dummy);
        Healpix_Map<double> gamma_in_res_map_1(in_nside, RING, dummy);
        Healpix_Map<double> gamma_in_res_map_2(in_nside, RING, dummy);

        alm2map_pol(gamma_Elms, gamma_Elms, gamma_Blms, gamma_in_res_map_T, gamma_in_res_map_1, gamma_in_res_map_2);

        Healpix_Map<double> gamma_map_1(out_nside, RING, dummy);
        Healpix_Map<double> gamma_map_2(out_nside, RING, dummy);
        if(out_nside < in_nside) {
            gamma_map_1.Import_degrade(gamma_in_res_map_1);
            gamma_map_2.Import_degrade(gamma_in_res_map_2);
        }else if(out_nside > in_nside) {
            gamma_map_1.Import_upgrade(gamma_in_res_map_1);
            gamma_map_2.Import_upgrade(gamma_in_res_map_2);
        }else {
            gamma_map_1 = gamma_in_res_map_1;
            gamma_map_2 = gamma_in_res_map_2;
        }

        Shears shears;
        shears.gamma1 = new double[out_npix];
        shears.gamma2 = new double[out_npix];
        for (int i = 0; i < out_npix; ++i) {
            shears.gamma1[i] = gamma_map_1[i];
            shears.gamma2[i] = gamma_map_2[i];
        }

        return shears;
    }

    double *shear2conv(int in_nside, int out_nside, Shears shears, int lmax){
        int in_npix = 12 * in_nside * in_nside;
        int out_npix = 12 * out_nside * out_nside;

        const nside_dummy dummy;
        arr<double> gamma_arr_1(shears.gamma1, in_npix);
        arr<double> gamma_arr_2(shears.gamma2, in_npix);
        Healpix_Map<double> gamma_map_T(in_nside, RING, dummy);
        Healpix_Map<double> gamma_map_1(gamma_arr_1, RING);
        Healpix_Map<double> gamma_map_2(gamma_arr_2, RING);
        gamma_map_T.fill(0);

        Alm<xcomplex<double>> gamma_Tlms(lmax, lmax);
        Alm<xcomplex<double>> gamma_Elms(lmax, lmax);
        Alm<xcomplex<double>> gamma_Blms(lmax, lmax);
        arr<double> weights(2 * in_nside, 1);

        map2alm_pol_iter(gamma_map_T, gamma_map_1, gamma_map_2, gamma_Tlms, gamma_Elms, gamma_Blms, 0, weights);

        Alm<xcomplex<double>> kappa_alms(lmax, lmax);

        xcomplex<double> *klm;
        xcomplex<double> *glm;
        klm = kappa_alms.mstart(0);
        *klm = 0;
        *(klm + 1) = 0;
        klm = kappa_alms.mstart(1);
        *(klm + 1) = 0;
        for (int l = 2; l <= lmax; ++l) {
            for (int m = 0; m <= l; ++m) {
                klm = kappa_alms.mstart(m) + l;
                glm = gamma_Elms.mstart(m) + l;

                *klm = -sqrt((double)l * (l + 1) / ((l + 2) * (l - 1))) * (*glm);
            }
        }

        Healpix_Map<double> kappa_in_res_map(in_nside, RING, dummy);

        alm2map(kappa_alms, kappa_in_res_map);

        Healpix_Map<double> kappa_map(out_nside, RING, dummy);
        if(out_nside < in_nside) {
            kappa_map.Import_degrade(kappa_in_res_map);
        }else if(out_nside > in_nside){
            kappa_map.Import_upgrade(kappa_in_res_map);
        } else{
            kappa_map = kappa_in_res_map;
        }

        double *kappa = new double[out_npix];
        for (int i = 0; i < out_npix; ++i) {
            kappa[i] = kappa_map[i];
        }

        return kappa;
    }
}