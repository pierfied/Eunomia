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
    void extern_test(){
        cout << "This Worked!\n";
    }

    void test_alm(int lmax, int npix, double *raw_map){
        Alm<xcomplex<double>> alms(lmax, lmax);
        alms.SetToZero();

        arr<double> map_arr(raw_map, npix);
        Healpix_Map<double> map(map_arr, RING);

        arr<double> weights(npix, 1);
        map2alm(map, alms, weights);

        xcomplex<double>* neg_ptr = alms.mstart(0) + lmax;
        xcomplex<double>* zero_ptr = alms.mstart(1) + 1;
        xcomplex<double>* pos_ptr = alms.mstart(1);
        cout << neg_ptr << '\n';
        cout << *neg_ptr << '\n';
        cout << zero_ptr << '\n';
        cout << *zero_ptr << '\n';
        cout << pos_ptr << '\n';
        cout << *pos_ptr << '\n';
        cout << "\n\n" << zero_ptr + lmax << "\t\t" << pos_ptr << "\n";
    }

    Shears conv2shear(int npix, double *raw_map, int lmax){
        int nside = Healpix_Base::npix2nside(npix);
        int order = Healpix_Base::nside2order(nside);

        arr<double> map_arr(raw_map, npix);
        Healpix_Map<double> kappa_map(map_arr, RING);

        Alm<xcomplex<double>> kappa_alms(lmax, lmax);
        kappa_alms.SetToZero();
        arr<double> weights(npix, 1);
        map2alm(kappa_map, kappa_alms, weights);

        Alm<xcomplex<double>> gamma_Elms(lmax, lmax);
        Alm<xcomplex<double>> gamma_Blms(lmax, lmax);
        gamma_Elms.SetToZero();
        gamma_Blms.SetToZero();

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

        Healpix_Map<double> gamma_map_T(order, RING);
        Healpix_Map<double> gamma_map_1(order, RING);
        Healpix_Map<double> gamma_map_2(order, RING);

        alm2map_pol(gamma_Elms, gamma_Elms, gamma_Blms, gamma_map_T, gamma_map_1, gamma_map_2);

        Shears shears;
        shears.gamma1 = new double[npix];
        shears.gamma2 = new double[npix];
        const double *g1_arr = gamma_map_1.Map().begin();
        const double *g2_arr = gamma_map_2.Map().begin();
        for (int i = 0; i < npix; ++i) {
            shears.gamma1[i] = g1_arr[i];
            shears.gamma2[i] = g2_arr[i];
        }

        return shears;
    }
}