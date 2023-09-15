# -*- coding: utf-8 -*-

#get correlation coeff. between median errors and resolution, try also stats.kendalltau!!
#calc. resolutions in atmosphere and ocean

#convert all resolution lists (containing the number of grid-boxes or model levels in the atmosphere and ocean) to np.array
lonres_atm = np.array(lonres_atm)
latres_atm = np.array(latres_atm)
lonres_oc = np.array(lonres_oc)
latres_oc = np.array(latres_oc)
lev_atm = np.array(lev_atm)
lev_oc = np.array(lev_oc)

#calculate 2d and 3d number of grid boxes in the atmosphere and ocean
hres_atm = np.array(lonres_atm)*np.array(latres_atm)
hres_oc = np.array(lonres_oc)*np.array(latres_oc)
res_atm = hres_atm*np.array(lev_atm)
res_oc = hres_oc*np.array(lev_oc)
res_atm_oc = res_atm + res_oc

#get rid of nans; this is necessary because the "nan-policy" input parameter does not exist any more in recent versions of scipy.stats.pearsonr
nanind = np.logical_or(np.isnan(median_error), np.isnan(res_atm_oc))

#rho for atmospheric resolution
rho_lonres_atm = stats.pearsonr(lonres_atm[~nanind],median_error[~nanind])
rho_latres_atm = stats.pearsonr(latres_atm[~nanind],median_error[~nanind])
rho_hres_atm = stats.pearsonr(hres_atm[~nanind],median_error[~nanind])
rho_vres_atm = stats.pearsonr(lev_atm[~nanind],median_error[~nanind])
rho_res_atm = stats.pearsonr(res_atm[~nanind],median_error[~nanind])
#rho for oceanic resolution
rho_lonres_oc = stats.pearsonr(lonres_oc[~nanind],median_error[~nanind])
rho_latres_oc = stats.pearsonr(latres_oc[~nanind],median_error[~nanind])
rho_hres_oc = stats.pearsonr(hres_oc[~nanind],median_error[~nanind])
rho_vres_oc = stats.pearsonr(lev_oc[~nanind],median_error[~nanind])
rho_res_oc = stats.pearsonr(res_oc[~nanind],median_error[~nanind])
#rho for atmospheric + oceanic resolutoin
rho_res_atm_oc = stats.pearsonr(res_atm_oc[~nanind],median_error[~nanind])

#spear for atmospheric resolution
spear_lonres_atm = stats.spearmanr(lonres_atm,median_error,nan_policy='omit')
spear_latres_atm = stats.spearmanr(latres_atm,median_error,nan_policy='omit')
spear_hres_atm = stats.spearmanr(hres_atm,median_error,nan_policy='omit')
spear_vres_atm = stats.spearmanr(lev_atm,median_error,nan_policy='omit')
spear_res_atm = stats.spearmanr(res_atm,median_error,nan_policy='omit')
#spear for oceanic resolution
spear_lonres_oc = stats.spearmanr(lonres_oc,median_error,nan_policy='omit')
spear_latres_oc = stats.spearmanr(latres_oc,median_error,nan_policy='omit')
spear_hres_oc = stats.spearmanr(hres_oc,median_error,nan_policy='omit')
spear_vres_oc = stats.spearmanr(lev_oc,median_error,nan_policy='omit')
spear_res_oc = stats.spearmanr(res_oc,median_error,nan_policy='omit')
#spear for atmospheric + oceanic resolution
spear_res_atm_oc = stats.spearmanr(res_atm_oc,median_error,nan_policy='omit')

##spear for ecs
##spear for atmospheric resolution
#spear_ecs_lonres_atm = stats.spearmanr(lonres_atm,ecs)
#spear_ecs_latres_atm = stats.spearmanr(latres_atm,ecs)
#spear_ecs_hres_atm = stats.spearmanr(hres_atm,ecs)
#spear_ecs_vres_atm = stats.spearmanr(lev_atm,ecs)
#spear_ecs_res_atm = stats.spearmanr(res_atm,ecs)
##spear for oceanic resolution
#spear_ecs_lonres_oc = stats.spearmanr(lonres_oc,ecs)
#spear_ecs_latres_oc = stats.spearmanr(latres_oc,ecs)
#spear_ecs_hres_oc = stats.spearmanr(hres_oc,ecs)
#spear_ecs_vres_oc = stats.spearmanr(lev_oc,ecs)
#spear_ecs_res_oc = stats.spearmanr(res_oc,ecs)
##spear for atmospheric + oceanic resolution
#spear_ecs_res_atm_oc = stats.spearmanr(res_atm_oc,ecs)
##spear for median error
#spear_ecs_median_error = stats.spearmanr(median_error,ecs)

##spear for tcr
##spear for atmospheric resolution
#spear_tcr_lonres_atm = stats.spearmanr(lonres_atm,tcr)
#spear_tcr_latres_atm = stats.spearmanr(latres_atm,tcr)
#spear_tcr_hres_atm = stats.spearmanr(hres_atm,tcr)
#spear_tcr_vres_atm = stats.spearmanr(lev_atm,tcr)
#spear_tcr_res_atm = stats.spearmanr(res_atm,tcr)
##spear for oceanic resolution
#spear_tcr_lonres_oc = stats.spearmanr(lonres_oc,tcr)
#spear_tcr_latres_oc = stats.spearmanr(latres_oc,tcr)
#spear_tcr_hres_oc = stats.spearmanr(hres_oc,tcr)
#spear_tcr_vres_oc = stats.spearmanr(lev_oc,tcr)
#spear_tcr_res_oc = stats.spearmanr(res_oc,tcr)
##spear for atmospheric + oceanic resolution
#spear_tcr_res_atm_oc = stats.spearmanr(res_atm_oc,tcr)
##spear for median error
#spear_tcr_median_error = stats.spearmanr(median_error,tcr)

#tau for atmospheric resolution
tau_lonres_atm = stats.kendalltau(lonres_atm,median_error,nan_policy='omit')
tau_latres_atm = stats.kendalltau(latres_atm,median_error,nan_policy='omit')
tau_hres_atm = stats.kendalltau(hres_atm,median_error,nan_policy='omit')
tau_vres_atm = stats.kendalltau(lev_atm,median_error,nan_policy='omit')
tau_res_atm = stats.kendalltau(res_atm,median_error,nan_policy='omit')
#tau for oceanic resolution
tau_lonres_oc = stats.kendalltau(lonres_oc,median_error,nan_policy='omit')
tau_latres_oc = stats.kendalltau(latres_oc,median_error,nan_policy='omit')
tau_hres_oc = stats.kendalltau(hres_oc,median_error,nan_policy='omit')
tau_vres_oc = stats.kendalltau(lev_oc,median_error,nan_policy='omit')
tau_res_oc = stats.kendalltau(res_oc,median_error,nan_policy='omit')
#tau for atmospheric + oceanic resolutoin
tau_res_atm_oc = stats.kendalltau(res_atm_oc,median_error,nan_policy='omit')

