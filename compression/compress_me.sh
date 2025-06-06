#!/bin/bash

#compresses netcdf4 files with ncks
SRCDIR=${HOME}/NFSMETEOGALICIA/web1/SWEN/lambtypes/6h/historical/sh
TARDIR=${HOME}/NFSMETEOGALICIA/web1/SWEN/lambtypes/6h/historical/sh/compressed
RUNDIR=${SRCDIR}
mkdir ${TARDIR}
cd ${RUNDIR}

ncks -4 -L 1  ${SRCDIR}/wtseries_cnrm_cm6_1_hr_historical_r1i1p1f2_sh_1979_2005.nc ${TARDIR}/wtseries_cnrm_cm6_1_hr_historical_r1i1p1f2_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_cnrm_cm6_1_historical_r1i1p1f2_sh_1979_2005.nc ${TARDIR}/wtseries_cnrm_cm6_1_historical_r1i1p1f2_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_giss_e2_1_g_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_giss_e2_1_g_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_sam0_unicon_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_sam0_unicon_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_bcc_csm2_mr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_bcc_csm2_mr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_gfdl_cm4_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_gfdl_cm4_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r24i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r24i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_access13_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_access13_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_mr_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_mr_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_cmcc_cm_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_cmcc_cm_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_access10_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_access10_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ccsm4_historical_r6i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ccsm4_historical_r6i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth_historical_r12i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth_historical_r12i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_canesm2_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_canesm2_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_lr_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_lr_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_cnrm_cm5_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_cnrm_cm5_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_giss_e2_h_historical_r6i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_giss_e2_h_historical_r6i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_inm_cm4_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_inm_cm4_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc_esm_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_miroc_esm_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm1_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_mri_esm1_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm1_m_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_noresm1_m_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_mr_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_mr_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc5_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_miroc5_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm2_0_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mri_esm2_0_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_ham_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_ham_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_cnrm_esm2_1_historical_r1i1p1f2_sh_1979_2005.nc ${TARDIR}/wtseries_cnrm_esm2_1_historical_r1i1p1f2_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_access_cm2_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_access_cm2_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_access_esm1_5_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_access_esm1_5_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_cmcc_cm2_sr5_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_cmcc_cm2_sr5_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc_es2l_historical_r1i1p1f2_sh_1979_2005.nc ${TARDIR}/wtseries_miroc_es2l_historical_r1i1p1f2_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc_es2l_historical_r5i1p1f2_sh_1979_2005.nc ${TARDIR}/wtseries_miroc_es2l_historical_r5i1p1f2_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_veg_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_veg_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_veg_lr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_veg_lr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc6_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_miroc6_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_aerchem_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_aerchem_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_cc_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_cc_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r2i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r4i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r4i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r5i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r5i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r6i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r6i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r7i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r7i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r9i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r9i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r10i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r10i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm2_lm_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_noresm2_lm_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r14i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r14i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r16i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r16i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r17i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r17i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r18i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r18i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r19i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r19i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r20i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r20i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r21i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r21i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r22i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r22i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r15i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r15i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r23i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r23i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r24i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r24i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r25i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r25i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r32i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r32i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_hadgem2_es_historical_r2i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_hadgem2_es_historical_r2i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_lr_historical_r8i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_lr_historical_r8i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_access_esm1_5_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_access_esm1_5_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm2_mm_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_noresm2_mm_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r4i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_lr_historical_r4i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r10i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r10i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r11i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r11i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r12i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r12i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm6a_lr_historical_r13i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm6a_lr_historical_r13i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r10i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r10i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r2i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_lr_historical_r2i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r3i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_lr_historical_r3i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r5i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_lr_historical_r5i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r6i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_ipsl_cm5a_lr_historical_r6i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r6i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r6i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r7i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r7i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r8i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r8i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r9i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r9i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm2_0_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mri_esm2_0_historical_r2i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm2_0_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mri_esm2_0_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm2_0_historical_r5i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mri_esm2_0_historical_r5i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_fgoals_g2_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_fgoals_g2_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_fgoals_g3_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_fgoals_g3_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_kiost_esm_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_kiost_esm_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_iitm_esm_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_iitm_esm_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_taiesm1_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_taiesm1_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_csiro_mk3_6_0_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_csiro_mk3_6_0_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r5i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r5i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r2i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_mpi_esm_1_2_hr_historical_r4i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_mpi_esm_1_2_hr_historical_r4i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_gfdl_cm3_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_gfdl_cm3_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_giss_e2_r_historical_r6i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_giss_e2_r_historical_r6i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_awi_esm_1_1_lr_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_awi_esm_1_1_lr_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r3i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r4i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r4i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r7i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r7i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r10i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r10i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r12i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r12i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r14i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r14i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r16i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r16i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r17i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r17i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r18i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_ec_earth3_historical_r18i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_hadgem2_cc_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_hadgem2_cc_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_hadgem2_es_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/wtseries_hadgem2_es_historical_r1i1p1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_hadgem3_gc31_mm_historical_r1i1p1f3_sh_1979_2005.nc ${TARDIR}/wtseries_hadgem3_gc31_mm_historical_r1i1p1f3_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm2_lm_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_noresm2_lm_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm2_lm_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_noresm2_lm_historical_r2i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_nesm3_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/wtseries_nesm3_historical_r1i1p1f1_sh_1979_2005.nc
ncks -4 -L 1 ${SRCDIR}/wtseries_nesm3_historical_r5i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_mri_esm2_0_historical_r4i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_noresm2_mm_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ipsl_cm5a_lr_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_interim_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_jra55_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_cnrm_cm6_1_historical_r2i1p1f2_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_cnrm_cm6_1_historical_r3i1p1f2_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_cmcc_esm2_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_bcc_csm1_1_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_gfdl_esm4_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_miroc6_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_veg_historical_r6i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_gfdl_esm2g_historical_r1i1p1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_canesm5_historical_r1i1p2f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_inm_cm5_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_cmcc_cm2_hr4_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_kace_1_0_g_historical_r1i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_era5_historical_r1i1p1_sh_1979_2020.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_nesm3_historical_r2i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_nesm3_historical_r3i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_nesm3_historical_r4i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r19i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r20i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r21i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r23i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_ec_earth3_historical_r25i1p1f1_sh_1979_2005.nc ${TARDIR}/
ncks -4 -L 1 ${SRCDIR}/wtseries_era5cs_historical_r1i1p1_sh_1979_2020.nc ${TARDIR}/

