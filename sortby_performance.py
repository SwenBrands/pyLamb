# -*- coding: utf-8 -*-

#sorts the errors and all associated variables by the median error in ascending order, i.e. from best to worse
sortind = np.argsort(median_error)
median_error = median_error[sortind]
mean_error = mean_error[sortind]
arr_error = arr_error[:,:,sortind]
id_error = id_error[:,:,sortind]
arr_error_2d = arr_error_2d[:,sortind]
arr_error_2d_gcms = arr_error_2d_gcms[:,sortind]
model = list(np.array(model)[sortind])
mrun = list(np.array(mrun)[sortind])
model_plus_run = list(np.array(model_plus_run)[sortind])
model_plus_cmip = list(np.array(model_plus_cmip)[sortind])
model_plus_cmip_orig = list(np.array(model_plus_cmip_orig)[sortind])
doi = list(np.array(doi)[sortind])
atmos = list(np.array(atmos)[sortind])
surface = list(np.array(surface)[sortind])
ocean = list(np.array(ocean)[sortind])
seaice = list(np.array(seaice)[sortind])
aerosols = list(np.array(aerosols)[sortind])
chemistry = list(np.array(chemistry)[sortind])
obgc = list(np.array(obgc)[sortind])
landice = list(np.array(landice)[sortind])
complexity = list(np.array(complexity)[sortind])
addinfo = list(np.array(addinfo)[sortind])
family = list(np.array(family)[sortind])
cmip = cmip[sortind]
rgb = list(np.array(rgb)[sortind])
marker = list(np.array(marker)[sortind])
latres_atm = list(np.array(latres_atm)[sortind])
lonres_atm = list(np.array(lonres_atm)[sortind])
latres_oc = list(np.array(latres_oc)[sortind])
lonres_oc = list(np.array(lonres_oc)[sortind])
lev_oc = list(np.array(lev_oc)[sortind])
ecs = list(np.array(ecs)[sortind])
tcr = list(np.array(tcr)[sortind])
experiment = list(np.array(experiment)[sortind])
