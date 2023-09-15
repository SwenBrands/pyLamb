#!/bin/bash

#This script compresses netcdf4 files with ncks
source ${HOME}/.bashrc
#set input parameters
SRCDIR=${LUSTRE}/datos/tareas/lamb_cmip5/results_v2/6h/historical/nh
TARDIR=${LUSTRE}/datos/tareas/lamb_cmip5/results_v2/6h/historical/nh/compressed
RUNDIR=${SRCDIR}

####EXECUTING###########################################################
cd ${RUNDIR}
mkdir ${TARDIR}
echo Cleaning ${TARDIR} ....
rm ${TARDIR}/wtseries*nc
echo copying files in ${SRCDIR}/ to ${TARDIR} ....
cp ${SRCDIR}/wtseries*nc ${TARDIR}

for file in ${TARDIR}/*
do
    echo processing ${file}
    ncks -4 -L 1 -O ${file} ${file}
done


