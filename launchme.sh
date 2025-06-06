#!/bin/bash

# send to queue with e.g.:
# sbatch --job-name=skill_decadal --partition=meteo_long --mem=28G --time=04:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 launchme.sh &
# load your software
source ${HOME}/.bashrc

#input parameters
mode='decadal_skill_maps' #set script to be run. Either 'interpolate', 'makecalcs', 'analyse', 'get_csd_local', 'map_lowfreq_var' or 'decadal_skill_maps'

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
	python makecalcs_parallel_plus_counts.py > ${LOGDIR}/log_makecalcs_parallel_plus_counts.log
elif [ ${mode} = 'analyse' ]
then
	python analysis_hist.py > ${LOGDIR}/log_analysis_hist.log
elif [ ${mode} = 'map_lowfreq_var' ]
then
	python map_lowfreq_var.py > ${LOGDIR}/map_lowfreq_var.log
elif [ ${mode} = 'get_csd_local' ]
then
	python get_csd_local.py > ${LOGDIR}/get_csd_local.log
elif [ ${mode} = 'decadal_skill_maps' ]
then
	python skill_maps_from_mon_counts.py > ${LOGDIR}/skill_maps_from_mon_counts.log
else
	echo 'Unknown entry mode=${mode}, exiting now....'
fi

echo "launme.sh has been sent to queue successfully, exiting now..."
exit()
