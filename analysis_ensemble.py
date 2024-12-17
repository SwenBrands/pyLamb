# -*- coding: utf-8 -*-
"""
This script is similar to analysis_hist.py but performs a multi GCM member analysis as specified in get_ensemble_metadata.py

@author: Swen Brands, swen.brands@gmail.com
"""
#compare the frequency of lamb types (period 1979-2005) coming from both observational and models data

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr
from scipy import stats
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.basemap import addcyclic
import os
exec(open('ntaylor.py').read())
exec(open('get_ensemble_metadata.py').read())
exec(open('analysis_functions.py').read())

#results for the following models are problematic and have to be revised
#model = ['noresm2_mm']
#mrun = ['r2i1p1f1']

filesystem = 'lustre' #set the filesystem in use, currently, extdisk or lustre
taryears = ['1979', '2005'] #start and end years of the performance analysis, for catalogues available from 1979 to 2005, i.e. the common period of the historical experiments from CMIP5 and 6
alt_taryears=['1850', '2014'] #alternative start and end years, used for those catalogues available from 1850 to 2014
alt_taryears_2=['1979', '2014'] ##alternative start and end years, used for those catalogues available from 1979 to 2014
fliersize = 0.5
textsize = 2. #3 for visualization without the 50 CESM2 members
textsize2 = 5.
figformat = 'pdf'
dpival = 300
classes_needed = 27 #20 for southern hemisphere
minfreq = 0.001
snapval = True #to be used in pcolormesh
errortype = 'MAE' #MAE or KL
region = 'cordexna' #nh, sh, escena, both, eurocordex or cordexna
cbounds_mae = np.linspace(0,1.7,18)
cbounds_kl = np.linspace(0,0.2,21)
#colormap1 = crea_cmap(cbounds1, rgbs, under, over)
colormap_best = 'jet'
colormap_error = 'jet' #nipy_spectral
upperlim = 3.0 #upper limit for summary MAE boxplot's Y-axis, 3.5 for NH
experiment = 'historical'
refdata = 'era5' #interim, jra55 or era5
flierprops = dict(marker='+', markerfacecolor='black', markersize=6, linestyle='none')
showcaps = False
showfliers = False #False or True
showmeans = False
rotation = 90. #rotation of the model labels in degrees
sigma_lim = 1.55 #1.55, limit of standard deviation ratio used in Taylor plot
figfolder = 'figs_ensemble'
correct_ru = 'no'
rank_ru = 0
orientation = 'h' #v or h; orientation of the bars in the boxplot
boxplot_linewidth = 0.5
aspect_ratio = 0.05
shrinkfactor = 0.6

##join all ensembles
# model = access_esm1_5_ens+hadgem2_es_ens+mpi_esm_1_2_lr_ens+mpi_esm_1_2_hr_ens+nesm3_ens+noresm2_lm_ens+noresm2_mm_ens+ec_earth3_ens+cnrm_cm6_1_ens+ipsl_cm5a_lr_ens+ipsl_cm6a_lr_ens+miroc_es2l_ens+mri_esm2_0_ens
# experiment = access_esm1_5_exp+hadgem2_es_exp+mpi_esm_1_2_lr_exp+mpi_esm_1_2_hr_exp+nesm3_exp+noresm2_lm_exp+noresm2_mm_exp+ec_earth3_exp+cnrm_cm6_1_exp+ipsl_cm5a_lr_exp+ipsl_cm6a_lr_exp+miroc_es2l_exp+mri_esm2_0_exp
# mrun = access_esm1_5_mrun+hadgem2_es_mrun+mpi_esm_1_2_lr_mrun+mpi_esm_1_2_hr_mrun+nesm3_mrun+noresm2_lm_mrun+noresm2_mm_mrun+ec_earth3_mrun+cnrm_cm6_1_mrun+ipsl_cm5a_lr_mrun+ipsl_cm6a_lr_mrun+miroc_es2l_mrun+mri_esm2_0_mrun
# family = access_esm1_5_family+hadgem2_es_family+mpi_esm_1_2_lr_family+mpi_esm_1_2_hr_family+nesm3_family+noresm2_lm_family+noresm2_mm_family+ec_earth3_family+cnrm_cm6_1_family+ipsl_cm5a_lr_family+ipsl_cm6a_lr_family+miroc_es2l_family+mri_esm2_0_family
# rgb = access_esm1_5_rgb+hadgem2_es_rgb+mpi_esm_1_2_lr_rgb+mpi_esm_1_2_hr_rgb+nesm3_rgb+noresm2_lm_rgb+noresm2_mm_rgb+ec_earth3_rgb+cnrm_cm6_1_rgb+ipsl_cm5a_lr_rgb+ipsl_cm6a_lr_rgb+miroc_es2l_rgb+mri_esm2_0_rgb
# cmip = access_esm1_5_cmip+hadgem2_es_cmip+mpi_esm_1_2_lr_cmip+mpi_esm_1_2_hr_cmip+nesm3_cmip+noresm2_lm_cmip+noresm2_mm_cmip+ec_earth3_cmip+cnrm_cm6_1_cmip+ipsl_cm5a_lr_cmip+ipsl_cm6a_lr_cmip+miroc_es2l_cmip+mri_esm2_0_cmip
# marker = access_esm1_5_marker+hadgem2_es_marker+mpi_esm_1_2_lr_marker+mpi_esm_1_2_hr_marker+nesm3_marker+noresm2_lm_marker+noresm2_mm_marker+ec_earth3_marker+cnrm_cm6_1_marker+ipsl_cm5a_lr_marker+ipsl_cm6a_lr_marker+miroc_es2l_marker+mri_esm2_0_marker

model = cesm2_ens + mpi_esm_1_2_lr_ens + mpi_esm_1_2_hr_ens + nesm3_ens + noresm2_lm_ens + noresm2_mm_ens + ec_earth3_ens + ec_earth3_veg_ens + cnrm_cm6_1_ens + ipsl_cm5a_lr_ens + ipsl_cm6a_lr_ens + miroc6_ens + miroc_es2l_ens + mri_esm2_0_ens
experiment = cesm2_exp + mpi_esm_1_2_lr_exp + mpi_esm_1_2_hr_exp + nesm3_exp + noresm2_lm_exp + noresm2_mm_exp + ec_earth3_exp + ec_earth3_veg_exp + cnrm_cm6_1_exp + ipsl_cm5a_lr_exp + ipsl_cm6a_lr_exp + miroc6_exp + miroc_es2l_exp + mri_esm2_0_exp
mrun = cesm2_mrun + mpi_esm_1_2_lr_mrun + mpi_esm_1_2_hr_mrun + nesm3_mrun + noresm2_lm_mrun + noresm2_mm_mrun + ec_earth3_mrun + ec_earth3_veg_mrun + cnrm_cm6_1_mrun + ipsl_cm5a_lr_mrun + ipsl_cm6a_lr_mrun + miroc6_mrun + miroc_es2l_mrun + mri_esm2_0_mrun
family = cesm2_family + mpi_esm_1_2_lr_family + mpi_esm_1_2_hr_family + nesm3_family + noresm2_lm_family + noresm2_mm_family + ec_earth3_family + ec_earth3_veg_family + cnrm_cm6_1_family + ipsl_cm5a_lr_family + ipsl_cm6a_lr_family + miroc6_family + miroc_es2l_family + mri_esm2_0_family
rgb = cesm2_rgb + mpi_esm_1_2_lr_rgb + mpi_esm_1_2_hr_rgb + nesm3_rgb + noresm2_lm_rgb + noresm2_mm_rgb + ec_earth3_rgb + ec_earth3_veg_rgb + cnrm_cm6_1_rgb + ipsl_cm5a_lr_rgb + ipsl_cm6a_lr_rgb + miroc6_rgb + miroc_es2l_rgb + mri_esm2_0_rgb
cmip = cesm2_cmip + mpi_esm_1_2_lr_cmip + mpi_esm_1_2_hr_cmip + nesm3_cmip + noresm2_lm_cmip + noresm2_mm_cmip + ec_earth3_cmip + ec_earth3_veg_cmip + cnrm_cm6_1_cmip + ipsl_cm5a_lr_cmip + ipsl_cm6a_lr_cmip + miroc6_cmip + miroc_es2l_cmip + mri_esm2_0_cmip
marker = cesm2_marker + mpi_esm_1_2_lr_marker + mpi_esm_1_2_hr_marker + nesm3_marker + noresm2_lm_marker + noresm2_mm_marker + ec_earth3_marker + ec_earth3_veg_marker + cnrm_cm6_1_marker + ipsl_cm5a_lr_marker + ipsl_cm6a_lr_marker + miroc6_marker + miroc_es2l_marker + mri_esm2_0_marker
gcm_name = cesm2_name + mpi_esm_1_2_lr_name + mpi_esm_1_2_hr_name + nesm3_name + noresm2_lm_name + noresm2_mm_name + ec_earth3_name + ec_earth3_veg_name + cnrm_cm6_1_name + ipsl_cm5a_lr_name + ipsl_cm6a_lr_name + miroc6_name + miroc_es2l_name + mri_esm2_0_name

# model = ec_earth3_ens+ec_earth3_veg_ens
# experiment = ec_earth3_exp+ec_earth3_veg_exp
# mrun = ec_earth3_mrun+ec_earth3_veg_mrun
# family = ec_earth3_family+ec_earth3_veg_family
# rgb = ec_earth3_rgb+ec_earth3_veg_rgb
# cmip = ec_earth3_cmip+ec_earth3_veg_cmip
# marker = ec_earth3_marker+ec_earth3_veg_marker
# gcm_name = ec_earth3_name+ec_earth3_veg_name

taylorprop = np.concatenate((np.expand_dims(np.array(marker),1),np.expand_dims(np.array(rgb),1),np.expand_dims(np.array(rgb),1)),axis=1)
taylorprop[:,2] = 'black'

##EXECUTE###############################################################
home = os.getenv('HOME')
#root to input netcdf files
if filesystem == 'extdisk': #for external hard disk
    tarpath = '/media/swen/ext_disk2/datos/lamb_cmip5/results/6h' #path of the source files
    figpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/figs'
    auxpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/pyLamb/aux'
elif filesystem == 'lustre': #for filesystems at IFCA
    tarpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/6h' #path of the source files
    #tarpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results/6h'
    figpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs'
    auxpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb/aux'
else:
    raise Exception('ERROR: unknown entry for <filesystem>!')

print('INFO: starting analysis for '+region+', '+experiment[0]+', ' +refdata+', '+errortype)
print('INFO: output figures will be saved in '+figfolder+' as '+figformat)
print('INFO: minimum frequency is '+str(minfreq)+' for '+str(classes_needed)+' LWTs')
print('INFO: target years are '+taryears[0]+' to '+taryears[1])

#create output directories if necessary
if os.path.isdir(figpath+'/'+figfolder+'/'+region+'/maps') != True:
    os.makedirs(figpath+'/'+figfolder+'/'+region+'/maps')
if os.path.isdir(figpath+'/'+figfolder+'/'+region+'/taylor') != True:
    os.makedirs(figpath+'/'+figfolder+'/'+region+'/taylor')

csvfile = figpath+'/'+figfolder+'/'+region+'/'+errortype+'_ensemble_ref_'+refdata+'_'+region+'.csv' #path of the csv file containing the median error for each model
yamlfile = figpath+'/'+figfolder+'/'+region+'/'+errortype+'_ensemble_ref_'+refdata+'_'+region+'.yaml' #path of the csv file containing the median error for each model

if region in ('nh','eurocordex','cordexna','escena'):
    hemis = 'nh'
elif region in ('sh','samerica'):
    hemis = 'sh'
else:
    raise Exception('ERROR: check entry for <region>')

if errortype == 'MAE':
    cbounds_map = cbounds_mae
    errorunit = '%'
elif errotype == 'KL':
    cbounds_mas = cbounds_kl
    errorunit = 'entropy'
else:
    raise Exception('ERROR: check entry for <errortype>!')
#model_plus_run = [model[ii]+' '+mrun[ii] for ii in range(len(model))]
model_plus_run = [gcm_name[ii]+' '+mrun[ii] for ii in range(len(model))]
model_plus_exp = [model[ii]+' '+experiment[ii][0] for ii in range(len(model))]
model_plus_cmip = [model[ii]+' '+str(int(cmip[ii])) for ii in range(len(model))]
cmip = [int(cmip[ii]) for ii in range(len(cmip))]
    
cbounds_ranking = list(np.arange(0.5,len(model)+1,1))
ticks_ranking = range(1,len(model)+1)

#set the colormap for the ranking maps
colormap_ranking = plt.cm.jet #'Set3'
#extract all colors from this colormap
cmaplist = [colormap_ranking(i) for i in range(colormap_ranking.N)]
#convert to linearly segmented colormap
colormap_ranking = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, colormap_ranking.N)

##read and load observational data
if refdata == 'interim':
    obs_srcpath = tarpath+'/historical/'+hemis+'/wtseries_interim_historical_r1i1p1_'+hemis+'_1979_2005.nc'
elif refdata == 'jra55':
    obs_srcpath = tarpath+'/historical/'+hemis+'/wtseries_jra55_historical_r1i1p1_'+hemis+'_1979_2005.nc'
elif refdata == 'era5':
    obs_srcpath = tarpath+'/historical/'+hemis+'/wtseries_era5_historical_r1i1p1_'+hemis+'_1979_2022.nc'
else:
    raise Exception('ERROR: unknown entry for <refdata>!')

#get <latlims> and <lonlims>
latlims, lonlims = get_target_coords(region)

obs_dataset = xr.open_dataset(obs_srcpath)
#shift from 0-360 to -180 to 180 and reassign values < 0 to lons > 180
ind_roll = np.argmin(np.abs(obs_dataset.lon.values-180))-1
obs_dataset = obs_dataset.roll(lon=ind_roll,roll_coords=True)
newlons = obs_dataset.lon.values
newlons[np.where(newlons > 180.)] = newlons[np.where(newlons > 180.)]-360.
obs_dataset.assign_coords(lon=newlons)

#cut out the target latitudes
latind = np.where((obs_dataset.lat >= latlims[0]) & (obs_dataset.lat <= latlims[1]))[0]
lonind = np.where((obs_dataset.lon >= lonlims[0]) & (obs_dataset.lon <= lonlims[1]))[0]
obs_dataset = obs_dataset.isel(lat = latind, lon = lonind)

lats = obs_dataset.variables['lat']
lats_values = lats.values
#lats_values = lats_values - np.diff(lats_values)[0]/2
halfres = np.abs(np.diff(lats_values)[0])/2
lons = obs_dataset.variables['lon']
lons_values = lons.values
#lons_values[np.where(lons_values>=180)] = lons_values[np.where(lons_values>=180)]-360
#lons_values[np.where(lons_values>180)] = lons_values[np.where(lons_values>180)]-360
#lons_values = lons_values - np.diff(lons_values)[0]/2

#filter out target years in obs
dates_obs = pd.to_datetime(obs_dataset.variables['time'].values)
years_ind_obs = ((dates_obs.year >= int(taryears[0])) & (dates_obs.year <= int(taryears[1])))
obs_dataset = obs_dataset.isel(time = years_ind_obs)
dates_obs = dates_obs[years_ind_obs]

#dates_obs_pd = pd.to_datetime(dates.values)
obs_wt = obs_dataset.variables['wtseries'].values
obs_dataset.close()

#get copies for hadgem2 models and get rid of December 2005 data (which is lacking for these models)
obs_wt_hadgem = np.copy(obs_wt)
dates_obs_hadgem = dates_obs.copy()
#dates_obs_pd_hadgem = dates_obs_pd.copy(deep=True)
#outind = np.where((dates_obs_pd_hadgem.month==12) & (dates_obs_pd_hadgem.year==2005))[0]
outind = np.where((dates_obs_hadgem.month==12) & (dates_obs_hadgem.year==2005))[0]
obs_wt_hadgem = np.delete(obs_wt_hadgem,outind,axis=0)
dates_obs_hadgem = dates_obs_hadgem.delete(outind)
#del(dates_pd_hadgem)
#dates_pd_hadgem = pd.to_datetime(dates_hadgem)

#calculate relative frequencies (obs)
dim_lon = len(lons)
dim_lat = len(lats)
dim_t = len(dates_obs) #time dimension for obs data, note that time dimension of the models may differ.
dim_t_hadgem = len(dates_obs_hadgem)
dim_wt = 27
obs_count_matrix = np.zeros((dim_wt, dim_lon, dim_lat))
obs_count_matrix_hadgem = np.zeros((dim_wt, dim_lon, dim_lat))
for i in range(dim_lon):
    for j in range(dim_lat):
        obs_wtcounts = np.histogram(obs_wt[:,i,j], bins=range(1,dim_wt+2)) #contains the number of times that each lamb type appears
        obs_wtcounts_hadgem = np.histogram(obs_wt_hadgem[:,i,j], bins=range(1,dim_wt+2)) #repeat for hadgem models
        obs_count_matrix[:,i,j] = obs_wtcounts[0]
        obs_count_matrix_hadgem[:,i,j] = obs_wtcounts_hadgem[0]
obs_freq = obs_count_matrix / dim_t
obs_freq_hadgem = obs_count_matrix_hadgem / dim_t_hadgem

#get grid boxes and Lamb circulation types that are lower than the minimum frequency criterium defined in minfreq
val_mat = np.zeros(obs_count_matrix.shape)+1
out_ind = np.where(obs_freq < minfreq)
val_mat[out_ind] = 0

#Calculate the error
arr_error = np.zeros((dim_lon,dim_lat,len(model)))
arr_mod_freq = np.zeros((dim_wt,dim_lon,dim_lat,len(model)))
for mm in list(range(len(model))):
    print('info: start validating '+model[mm]+', '+experiment[mm].upper()+' and '+mrun[mm])
    if (model[mm] == 'ec_earth3_veg') & (mrun[mm] in ('r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1','r11i1p1f1')) or (model[mm] == 'mpi_esm_1_2_hr') & (mrun[mm] in ('r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1')):
        print('INFO: The LWT catalogue for '+model[mm]+', '+experiment[mm]+', '+mrun[mm]+' is available from '+str(alt_taryears[0])+' to '+str(alt_taryears[1]))
        mod_srcpath = tarpath+'/'+experiment[mm]+'/'+hemis+'/wtseries_'+model[mm]+'_'+experiment[mm]+'_'+mrun[mm]+'_'+hemis+'_'+alt_taryears[0]+'_'+alt_taryears[1]+'.nc'
    elif model[mm] in ('cesm2','cnrm_cm6_1','ec_earth3','ipsl_cm6a_lr','miroc6','miroc_es2l','mpi_esm_1_2_lr','mri_esm2_0','nesm3','noresm2_lm','noresm2_mm'):
        print('INFO: The LWT catalogue for '+model[mm]+', '+experiment[mm]+', '+mrun[mm]+' is available from '+str(alt_taryears_2[0])+' to '+str(alt_taryears_2[1]))
        mod_srcpath = tarpath+'/'+experiment[mm]+'/'+hemis+'/wtseries_'+model[mm]+'_'+experiment[mm]+'_'+mrun[mm]+'_'+hemis+'_'+alt_taryears_2[0]+'_'+alt_taryears_2[1]+'.nc'
    else:
        mod_srcpath = tarpath+'/'+experiment[mm]+'/'+hemis+'/wtseries_'+model[mm]+'_'+experiment[mm]+'_'+mrun[mm]+'_'+hemis+'_'+taryears[0]+'_'+taryears[1]+'.nc'
    #open model dataset
    mod_dataset = xr.open_dataset(mod_srcpath)
    
    #cut out target years
    dates_mod = pd.to_datetime(mod_dataset.variables['time'].values)
    years_ind_mod = ((dates_mod.year >= int(taryears[0])) & (dates_mod.year <= int(taryears[1])))
    mod_dataset = mod_dataset.isel(time = years_ind_mod)
    
    #shift from 0-360 to -180 to 180
    ind_roll = np.argmin(np.abs(mod_dataset.lon.values-180))-1
    mod_dataset = mod_dataset.roll(lon=ind_roll,roll_coords=True)
    newlons = mod_dataset.lon.values
    newlons[np.where(newlons > 180.)] = newlons[np.where(newlons > 180.)]-360.
    mod_dataset.assign_coords(lon=newlons)

    #retain grid boxes at a the lats defined in <latlims>
    latind = np.where((mod_dataset.lat >= latlims[0]) & (mod_dataset.lat <= latlims[1]))[0]
    lonind = np.where((mod_dataset.lon >= lonlims[0]) & (mod_dataset.lon <= lonlims[1]))[0]
    mod_dataset = mod_dataset.isel(lat = latind, lon = lonind)
    
    mod_wt = mod_dataset.variables['wtseries'][:]
    mod_wt = mod_wt.values
    #mod_wt = mod_wt.astype(np.int64) #convert to integer values
    mod_dates = mod_dataset.variables['time'][:]
    mod_dim_t = len(mod_dates)
    mod_dataset.close()
    
    #calculate relative frequencies (models)
    mod_count_matrix = np.zeros((dim_wt, dim_lon, dim_lat))
    for ii in range(dim_lon):
        for jj in range(dim_lat):
            mod_wtcounts = np.histogram(mod_wt[:,ii,jj],bins=range(1,dim_wt+2))
            mod_count_matrix[:,ii,jj] = mod_wtcounts[0]
    mod_freq = mod_count_matrix / mod_dim_t
    
    #save relative frecuencies of the model to data array
    arr_mod_freq[:,:,:,mm] = mod_freq 
    
    ##calculate the mean absolute error between models and observations 
    #error = 1 / np.float(dim_wt) * sum(abs(mod_freq - obs_freq))
    if errortype == 'MAE':
        if model[mm] in ('hadgem2_es','hadgem2_cc'):
            print('INFO: model is '+model[mm]+' , December 2005 data was removed...')
            error = np.mean(np.abs(mod_freq*100 - obs_freq_hadgem*100),axis=0)
        else:
            error = np.mean(np.abs(mod_freq*100 - obs_freq*100),axis=0)
    elif errortype == 'KL':
        if model[mm] in ('hadgem2_es','hadgem2_cc'):
            print('INFO: model is '+model[mm]+' , December 2005 data was removed...')
            error = stats.entropy(pk=mod_freq, qk=obs_freq_hadgem)
        else:
            error = stats.entropy(pk=mod_freq, qk=obs_freq)
    else:
        raise Exception('error: check entry for <errortype>!')
    arr_error[:,:,mm] = error

#take into account reanalysis uncertainty, see ananlysis_hist.py for infos on how to construct the auxiliary files <rean_file>
if correct_ru == 'yes':
    if (region == 'nh' and errortype == 'MAE' and classes_needed == 27 and minfreq == 0.001) or (region == 'sh' and errortype == 'MAE' and classes_needed == 20 and minfreq == 0.001):
        print('INFO: model error and ranks where ERA-Interim vs. JRA-55 not ranks first will be set to nan!')
        rean_file = auxpath+'/rank_'+errortype+'_interim_jra55_'+region+'_1979_2005.nc'
        corrnc = xr.open_dataset(rean_file) #contains n replicas of the matrix along the third axis, with n = number of models
        rean_nanmask = np.where(corrnc.rank_interim_vs_jra55_in_ensemble.values > rank_ru)
        arr_error[rean_nanmask] = np.nan
    else:
        raise Exception('ERROR: <correct_ru> is not defined for '+region+', '+errortype+', '+str(classes_needed)+', '+str(minfreq))
else:
    print('INFO: reanalysis uncertainty is not taken into account. If you like to filter out those grid-boxes where this kind of uncertainty is enhanced, set <correct_ru> to yes and re-run this script.')

##find performance rank of each model at each grid box
id_error = np.zeros(arr_error.shape)
for kk in list(range(arr_error.shape[0])):
        for zz in list(range(arr_error.shape[1])):
            #id_error[kk,zz,:] = np.argsort(arr_error[kk,zz,:])
            temp = np.argsort(arr_error[kk,zz,:])
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(arr_error[kk,zz,:]))
            id_error[kk,zz,:] = ranks
            
#id_error = np.argsort(arr_error,axis=2).astype('float')+1

##find the best performing model at each grid box
id_best = np.argmin(arr_error,axis=2)+1. #best rank is 1

##set grid box where Lamb classificaiton cannot be applied to nan
nanmask = np.where(np.sum(val_mat,axis=0) < classes_needed)

for ii in range(arr_error.shape[2]):
    #get matrix for a given model
    arr_error_step = arr_error[:,:,ii]
    id_error_step = id_error[:,:,ii]
    
    #set nans to these matrices
    arr_error_step[nanmask] = np.nan
    id_error_step[nanmask] = np.nan
    
    #reassign modified matrices to 3d arrays
    arr_error[:,:,ii] = arr_error_step
    id_error[:,:,ii] = id_error_step
    
id_best[nanmask] = np.nan

#summary boxplot for MAE
arr_error_2d = arr_error.reshape(arr_error.shape[0]*arr_error.shape[1],arr_error.shape[2])
median_error = np.nanmedian(arr_error_2d,axis=0)

##used for boxplot
#color_dict = dict(zip(model_plus_cmip, rgb))
color_dict = dict(zip(mrun, rgb))

##boxplot with seaborn
fig = sns.boxplot(data=arr_error_2d,orient=orientation,fliersize=fliersize,palette=rgb,linewidth=boxplot_linewidth)
#plt.subplots_adjust(bottom=0.20) #see https://www.python-graph-gallery.com/192-about-matplotlib-margins
if orientation == 'v':
    #fig.set_xticklabels(model_plus_run,rotation=rotation,size=textsize) #model_plus_exp
    fig.set_xticklabels(mrun,rotation=rotation,size=textsize) #model_plus_exp
    fig.set_ylim(0,upperlim)
    fig.set_ylabel(errortype.upper()+' of relative LWT frequencies ('+errorunit+')', size=textsize+3.)
elif orientation == 'h':
    plt.gca().set_aspect(aspect_ratio)
    fig.set_yticklabels(model_plus_run,rotation=0,size=textsize) #model_plus_exp
    fig.set_xlim(0,upperlim)
    fig.set_xticks(ticks = np.arange(upperlim+1),labels=np.arange(upperlim+1),size=textsize2)
    fig.set_xlabel(errortype.upper()+' of relative LWT frequencies ('+errorunit+')', size=textsize2)
    axes = plt.gca()
    axes.set_aspect(aspect_ratio)
else:
    raise Exception('ERROR: check entry for <orientation> !')

savepath = figpath+'/'+figfolder+'/'+region+'/boxplot_'+errortype+'_ensemble_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_1979-2005.'+figformat
plt.savefig(savepath, dpi=dpival)
plt.close('all')

#plot errors and ranking for each model
for mm in range(len(model)): 
    #error map
    fig = plt.figure()
    mymap = get_projection(region,lats_values,lons_values)
    plotme = np.transpose(arr_error[:,:,mm])
    XX,YY = np.meshgrid(lons_values-halfres,lats_values) #longitudes must be corrected to obtain cells centered at <lons_values>
    X, Y = mymap(XX, YY)
    plotme = np.ma.masked_where(np.isnan(plotme),plotme)
    mymap.pcolormesh(X, Y, plotme, cmap=colormap_error, latlon=False, snap=snapval)
    mymap.drawcoastlines()
    savepath = figpath+'/'+figfolder+'/'+region+'/maps/'+errortype+'_'+model[mm]+'_'+mrun[mm]+'_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_1979-2005.'+figformat
    cbar = plt.colorbar(shrink=shrinkfactor)
    plt.title(errortype+', '+model[mm]+', '+mrun[mm]+' w.r.t. '+refdata+', 1979-2005, all seasons') 
    plt.savefig(savepath, dpi=dpival)
    plt.close('all')
    
    #ID map
    fig = plt.figure()
    mymap = get_projection(region,lats_values,lons_values)
    if region in ('nh','sh'):
        plotme, lons_plot = addcyclic(np.transpose(id_error[:,:,mm]),lons_values)
    else:
        plotme = np.transpose(arr_error[:,:,mm])
        lons_plot = lons_values

    XX,YY = np.meshgrid(lons_plot,lats_values)
    X, Y = mymap(XX, YY)
    plotme = np.ma.masked_where(np.isnan(plotme),plotme)
    #norm = colors.BoundaryNorm(boundaries=cbounds_ranking, ncolors=len(model))
    norm = mpl.colors.BoundaryNorm(cbounds_ranking, colormap_ranking.N)
    mymap.pcolormesh(X, Y, plotme, cmap=colormap_ranking, norm = norm, latlon=False, snap=snapval)
    mymap.drawcoastlines()
    savepath = figpath+'/'+figfolder+'/'+region+'/maps/rank_'+model[mm]+'_'+mrun[mm]+'_wrt '+refdata+'_'+region+'_ruout_'+correct_ru+'_1979-2005.'+figformat
    #cbar = plt.colorbar(orientation="horizontal",fraction=0.07)
    cbar = plt.colorbar(shrink=1.)
    cbar.set_ticks(ticks_ranking)
    plt.title('Performance rank for '+model[mm]+', '+mrun[mm]+' w.r.t. '+refdata+', 1979-2005, all seasons') 
    plt.savefig(savepath, dpi=dpival)
    plt.close('all')


##plot the Taylor diagrams, first get the LWT per model, then flatten data for each model and finally calc statistics and draw plot for this LWT
for ww in range(dim_wt):
    stat = np.zeros((arr_mod_freq.shape[3],6))
    for mm in range(len(model)):
        if model in ('hadgem2_es','hadgem2_cc'):
            obs_freq_step = obs_freq_hadgem[ww,:,:]
        else:
            obs_freq_step = obs_freq[ww,:,:]
        #then set nans
        obs_freq_step[nanmask] = np.nan
            
        arr_mod_freq_step = arr_mod_freq[ww,:,:,mm]
        arr_mod_freq_step[nanmask] = np.nan
            
        #then flatten both types of arrays and drop nans
        obs_freq_step = obs_freq_step.flatten()
        obs_freq_step = np.delete(obs_freq_step,np.where(np.isnan(obs_freq_step))[0])
        arr_mod_freq_step = arr_mod_freq_step.flatten()
        arr_mod_freq_step = np.delete(arr_mod_freq_step,np.where(np.isnan(arr_mod_freq_step))[0])
   
        #calculated statistics needed for taylor
        statstep = get_statn(obs_freq_step, arr_mod_freq_step)
        stat[mm,:] = statstep
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plot taylor
    ax = diagn(ax, stat, taylorprop, sigma_lim=sigma_lim)
    #ax.legend(mrun[0:-4])
    savepath = figpath+'/'+figfolder+'/'+region+'/taylor/LWT_'+str(ww+1)+'_wrt '+refdata+'_'+region+'_ruout_'+correct_ru+'_1979-2005.'+figformat
    plt.savefig(savepath, dpi=dpival)
    plt.close('all')

#save main results in csv format
main_results = pd.DataFrame(data=np.round(median_error,4), index=model_plus_run, columns=[errortype])
main_results['cmip'] = cmip
family_bin = np.array(family)
family_bin[np.where(np.array(family) == 'gcm')] = 0
family_bin[np.where(np.array(family) == 'esm')] = 1
main_results['esm'] = list(family_bin)
main_results['reference'] = list(np.tile(refdata,len(model)))
main_results.to_csv(csvfile,header=[errortype,'cmip','esm','reference'])
