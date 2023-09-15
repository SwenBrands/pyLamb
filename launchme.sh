#!/bin/bash

#send to queue with e.g.:
#qsub -N makecalcs -l walltime=10:00:00 -l mem=64gb -q himem -e error.log -o out.log -l nodes=1:ppn=12 launchme.sh
#load your software
source ${HOME}/.bashrc

#input parameters
mode='makecalcs' #set script to be run. Either 'interpolate', 'makecalcs', 'analyse',get_csd_local or 'map_lowfreq_var'

#check python version
echo "Your Python version is:"
python --version
#launch to queue on lustre
RUNDIR=/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb
LOGDIR=/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb/LOG
cd ${RUNDIR}

if [ ${mode} = 'interpolate' ]
then
	python interpolator_xesmf.py > ${LOGDIR}/log_interpolator_xesmf.log
elif [ ${mode} = 'makecalcs' ]
then
	python makecalcs_parallel.py > ${LOGDIR}/log_makecalcs_parallel.log
elif [ ${mode} = 'analyse' ]
then
	python analysis_hist.py > ${LOGDIR}/log_analysis_hist.log
elif [ ${mode} = 'map_lowfreq_var' ]
then
	python map_lowfreq_var.py > ${LOGDIR}/map_lowfreq_var.log
elif [ ${mode} = 'get_csd_local' ]
then
	python get_csd_local.py > ${LOGDIR}/get_csd_local.log
else
	echo 'Unknown entry mode=${mode}, exiting now....'
fi

echo "launme.sh has been sent to queue successfully, exiting now..."
exit()
