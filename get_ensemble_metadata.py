# -*- coding: utf-8 -*-

##get metadata for EC-Earth3 ensemble of historical exeriments, note: errors were found in 'r22i1p1f1'
ec_earth3_mrun = ['r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1']
#ec_earth3_mrun = ['r1i1p1f1','r3i1p1f1']
ec_earth3_ens = ['ec_earth3'] * len(ec_earth3_mrun)
ec_earth3_family = ['gcm'] * len(ec_earth3_mrun)
ec_earth3_cmip = ['6']*len(ec_earth3_mrun)
ec_earth3_rgb = ['#F5F576']*len(ec_earth3_mrun)
ec_earth3_marker = ['v']*len(ec_earth3_mrun)
ec_earth3_exp = ['historical'] * len(ec_earth3_mrun)
ec_earth3_name = ['EC-Earth3'] * len(ec_earth3_mrun)

##get metadata for EC-Earth3_Veg ensemble of historical exeriments
ec_earth3_veg_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1','r11i1p1f1']
#ec_earth3_veg_mrun = ['r1i1p1f1','r2i1p1f1']
ec_earth3_veg_ens = ['ec_earth3_veg'] * len(ec_earth3_veg_mrun)
ec_earth3_veg_family = ['gcm'] * len(ec_earth3_veg_mrun)
ec_earth3_veg_cmip = ['6']*len(ec_earth3_veg_mrun)
ec_earth3_veg_rgb = ['#fffdaf']*len(ec_earth3_veg_mrun)
ec_earth3_veg_marker = ['>']*len(ec_earth3_veg_mrun)
ec_earth3_veg_exp = ['historical'] * len(ec_earth3_veg_mrun)
ec_earth3_veg_name = ['EC-Earth3-Veg'] * len(ec_earth3_veg_mrun)

#get metadata for mpi_esm_1_2_hr ensemble of historical exeriments
mpi_esm_1_2_hr_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
mpi_esm_1_2_hr_ens = ['mpi_esm_1_2_hr'] * len(mpi_esm_1_2_hr_mrun)
mpi_esm_1_2_hr_family = ['esm'] * len(mpi_esm_1_2_hr_mrun)
mpi_esm_1_2_hr_cmip = ['6']*len(mpi_esm_1_2_hr_mrun)
mpi_esm_1_2_hr_rgb = ['#D1FFDB']*len(mpi_esm_1_2_hr_mrun)
mpi_esm_1_2_hr_marker = ['>']*len(mpi_esm_1_2_hr_mrun)
#mpi_esm_1_2_hr_exp = ['amip'] + ['historical'] * (len(mpi_esm_1_2_hr_mrun)-1)
mpi_esm_1_2_hr_exp = ['historical'] * len(mpi_esm_1_2_hr_mrun)
mpi_esm_1_2_hr_name = ['MPI-ESM1.2-HR'] * len(mpi_esm_1_2_hr_mrun)

#get metadata for mpi_esm_1_2_lr ensemble of historical exeriments
mpi_esm_1_2_lr_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
mpi_esm_1_2_lr_ens = ['mpi_esm_1_2_lr'] * len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_family = ['esm'] * len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_cmip = ['6']*len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_rgb = ['#00FF00']*len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_marker = ['v']*len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_exp = ['historical'] * len(mpi_esm_1_2_lr_mrun)
mpi_esm_1_2_lr_name = ['MPI-ESM1.2-LR'] * len(mpi_esm_1_2_lr_mrun)

#get metadata for mri_esm2_0 ensemble of historical exeriments
#mri_esm2_0_mrun = ['r1i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1']
mri_esm2_0_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1']
mri_esm2_0_ens = ['mri_esm2_0'] * len(mri_esm2_0_mrun)
mri_esm2_0_family = ['esm'] * len(mri_esm2_0_mrun)
mri_esm2_0_cmip = ['6']*len(mri_esm2_0_mrun)
mri_esm2_0_rgb = ['#B5651D']*len(mri_esm2_0_mrun)
mri_esm2_0_marker = ['v']*len(mri_esm2_0_mrun)
#mri_esm2_0_exp = ['amip'] + ['historical'] * (len(mri_esm2_0_mrun)-1)
mri_esm2_0_exp = ['historical'] * len(mri_esm2_0_mrun)
mri_esm2_0_name = ['MRI-ESM2.0'] * len(mri_esm2_0_mrun)

#get metadata for ipsl_cm5a_lr ensemble of historical exeriments
ipsl_cm5a_lr_mrun = ['r1i1p1','r2i1p1','r3i1p1','r4i1p1','r5i1p1','r6i1p1']
ipsl_cm5a_lr_ens = ['ipsl_cm5a_lr'] * len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_family = ['esm'] * len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_cmip = ['5']*len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_rgb = ['grey']*len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_marker = ['P']*len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_exp = ['historical'] * len(ipsl_cm5a_lr_mrun)
ipsl_cm5a_lr_name = ['IPSL-CM5A-LR'] * len(ipsl_cm5a_lr_mrun)

#get metadata for ipsl_cm6a_lr ensemble of historical exeriments
ipsl_cm6a_lr_mrun = ['r1i1p1f1','r10i1p1f1','r11i1p1f1','r12i1p1f1','r13i1p1f1','r14i1p1f1','r15i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r22i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1','r32i1p1f1']
ipsl_cm6a_lr_ens = ['ipsl_cm6a_lr'] * len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_family = ['esm'] * len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_cmip = ['6']*len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_rgb = ['#d3d3d3']*len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_marker = ['v']*len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_exp = ['historical'] * len(ipsl_cm6a_lr_mrun)
ipsl_cm6a_lr_name = ['IPSL-CM6A-LR'] * len(ipsl_cm6a_lr_mrun)

#get metadata for hadgem2_es ensemble of historical exeriments
hadgem2_es_mrun = ['r1i1p1','r2i1p1']
hadgem2_es_ens = ['hadgem2_es'] * len(hadgem2_es_mrun)
hadgem2_es_family = ['esm'] * len(hadgem2_es_mrun)
hadgem2_es_cmip = ['5']*len(hadgem2_es_mrun)
hadgem2_es_rgb = ['#FFA500']*len(hadgem2_es_mrun)
hadgem2_es_marker = ['P']*len(hadgem2_es_mrun)
hadgem2_es_exp = ['historical'] * len(hadgem2_es_mrun)
hadgem2_es_name = ['HadGEM2-ES'] * len(hadgem2_es_mrun)

#get metadata for access_esm1_5 ensemble of historical exeriments
access_esm1_5_mrun = ['r1i1p1f1','r3i1p1f1']
access_esm1_5_ens = ['access_esm1_5'] * len(access_esm1_5_mrun)
access_esm1_5_family = ['esm'] * len(access_esm1_5_mrun)
access_esm1_5_cmip = ['6']*len(access_esm1_5_mrun)
access_esm1_5_rgb = ['#0000FF']*len(access_esm1_5_mrun)
access_esm1_5_marker = ['<']*len(access_esm1_5_mrun)
access_esm1_5_exp = ['historical'] * len(access_esm1_5_mrun)
access_esm1_5_name = ['ACCESS-ESM1.5'] * len(access_esm1_5_mrun)

#get metadata for noresm2_lm ensemble of historical exeriments
noresm2_lm_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1']
noresm2_lm_ens = ['noresm2_lm'] * len(noresm2_lm_mrun)
noresm2_lm_family = ['esm'] * len(noresm2_lm_mrun)
noresm2_lm_cmip = ['6']*len(noresm2_lm_mrun)
noresm2_lm_rgb = ['#ff69b4']*len(noresm2_lm_mrun)
noresm2_lm_marker = ['v']*len(noresm2_lm_mrun)
noresm2_lm_exp = ['historical'] * len(noresm2_lm_mrun)
noresm2_lm_name = ['NorESM2-LM'] * len(noresm2_lm_mrun)

#get metadata for noresm2_mm ensemble of historical exeriments
noresm2_mm_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1']
noresm2_mm_ens = ['noresm2_mm'] * len(noresm2_mm_mrun)
noresm2_mm_family = ['esm'] * len(noresm2_mm_mrun)
noresm2_mm_cmip = ['6']*len(noresm2_mm_mrun)
noresm2_mm_rgb = ['#ffb6c1']*len(noresm2_mm_mrun)
noresm2_mm_marker = ['<']*len(noresm2_mm_mrun)
noresm2_mm_exp = ['historical'] * len(noresm2_mm_mrun)
noresm2_mm_name = ['NorESM2-MM'] * len(noresm2_mm_mrun)

#get metadata for miroc_es2l ensemble of historical exeriments
miroc_es2l_mrun = ['r1i1p1f2','r5i1p1f2']
miroc_es2l_ens = ['miroc_es2l'] * len(miroc_es2l_mrun)
miroc_es2l_family = ['esm'] * len(miroc_es2l_mrun)
miroc_es2l_cmip = ['6']*len(miroc_es2l_mrun)
miroc_es2l_rgb = ['#008080']*len(miroc_es2l_mrun)
miroc_es2l_marker = ['<']*len(miroc_es2l_mrun)
miroc_es2l_exp = ['historical'] * len(miroc_es2l_mrun)
miroc_es2l_name = ['MIROC-ES2L'] * len(miroc_es2l_mrun)

#get metadata for nesm3 ensemble of historical exeriments
nesm3_mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1']
nesm3_ens = ['nesm3'] * len(nesm3_mrun)
nesm3_family = ['esm'] * len(nesm3_mrun)
nesm3_cmip = ['5']*len(nesm3_mrun)
nesm3_rgb = ['white']*len(nesm3_mrun)
nesm3_marker = ['>']*len(nesm3_mrun)
nesm3_exp = ['historical'] * len(nesm3_mrun)
nesm3_name = ['NESM3'] * len(nesm3_mrun)

#get metadata for cnrm_cm6_1 ensemble of historical exeriments
cnrm_cm6_1_mrun = ['r1i1p1f2','r2i1p1f2','r3i1p1f2']
cnrm_cm6_1_ens = ['cnrm_cm6_1'] * len(cnrm_cm6_1_mrun)
cnrm_cm6_1_family = ['gcm'] * len(cnrm_cm6_1_mrun)
cnrm_cm6_1_cmip = ['6']*len(cnrm_cm6_1_mrun)
cnrm_cm6_1_rgb = ['red']*len(cnrm_cm6_1_mrun)
cnrm_cm6_1_marker = ['v']*len(cnrm_cm6_1_mrun)
cnrm_cm6_1_exp = ['historical'] * len(cnrm_cm6_1_mrun)
cnrm_cm6_1_name = ['CNRM-CM6-1'] * len(cnrm_cm6_1_mrun)
