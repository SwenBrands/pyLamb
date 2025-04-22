# -*- coding: utf-8 -*-
"""
This is the main script for performing the inter-model performance analysis in:


Brands, S. (2022): A circulation-based performance atlas of the CMIP5 and 6 models for regional climate studies in the Northern Hemisphere mid-to-high latitudes, Geosci. Model Dev., 15, 1375–1411, https://doi.org/10.5194/gmd-15-1375-2022. 
The script loads the Lamb Weather Type (LWT) catalogues obtained with <makecalcs.py> for the models indicated in <model> and compares them with the reanalysis indicated in
<refdata> in terms of the verification measure set in <errortype>. The script returns <region> specific polar stereographic maps for this type of error, the corresponding
model ranks, a summary boxplot and Taylor diagrams comparing the modelled and observed relative frequencies for each of the 27 LWTs. The median error for each model is saved in csv format.

@Author: Swen Brands, swen.brands@gmail.com
"""

#load the required packages
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr
from scipy import stats
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.basemap import addcyclic
import yaml
#from utiles import crea_cmap
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import os
import pdb #then type <pdb.set_trace()> at a given line in the code below
import seaborn as sns
plt.style.use('seaborn')

#load additional functions; a further function is loaded in lines 88 bis 94 depending on the input parameter  <alt_runs> set in line 74.
exec(open('ntaylor.py').read())
exec(open('analysis_functions.py').read())

##INPUT PARAMETERS to be set by the user ####################################
home = os.getenv('HOME')
filesystem = 'lustre' #set the filesystem in use, currently, extdisk or lustre
experiment = 'historical'
refdata = 'era5' #interim, jra55 or era5 reference reanlaysis dataset
errortype = 'MAE' #error type to be computed, MAE, KL, TPMS, PERS or PERSall: Mean Absolute Error, Kullback-Leibler divergence, Transition Probability Matrix score (full matrix) or only diagonal thereof (PERS) or error in sum of the diagonal (PERSall) indicating probabilities of persistence.
timelag = 1 #lag for calculating transition probabilities, six-hourly time step i.e. 1 refers to a lag of six hours, 2 to 12 hours etc. (this is the time index), 4 for work with UZAR
region = 'sh' #nh, sh, eurocordex, cordexna, escena, north_atlantic
tarpath = '/media/swen/ext_disk2/datos/lamb_cmip5/results_v2/6h' #path of the source files
figpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/figs'
auxpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/pyLamb/aux'
figfolder = 'figs_ref'+refdata #name of the folder containing the output figures

##set target months and years
# season = [6,7,8,9] #for study with UZAR
# season_label = 'JJAS' #for study with UZAR
season = np.arange(1,13,1).tolist()
season_label = 'year'
taryears = ['1979', '2005'] #start and end years as indicated in source nc file containing the LWT catalogues (hereafter: "source files"); this is the period the analyses are conducted for; currently coincides with the years available for cmip5
taryears_cmip6 = ['1979', '2014']
taryears_cmip6_long = ['1850', '2014'] #alternative start and end years, used for those catalogues available from 1850 to 2014
#set LWT classification options
classes_needed = 10 #minimum number of classes required to plot the result on a map, 27 for NH and 20 for SH, 18 following Fernández-Granja et al. 2023, Climate Dynamics, https://doi.org/10.1007%2Fs00382-022-06658-7
minfreq = 0.001 #minimum frequency required to plot the results on a map (in decimals), 0.001 in gmd-2020-418
#set format and resolution of the output figures
figformat = 'pdf' #format of output figures, pdf or png
dpival = 300 #resolution of output figures
#set mapping options
cbounds_mae = np.linspace(0,1.7,18) #colormap intervals for the MAE
cbounds_kl = np.linspace(0,0.2,21) #colormap intervals for the Kullback-Leibler divergence
cbounds_tpms = np.linspace(0,0.2,21) #colormap intervals for the TPMS in %
cbounds_pers = np.linspace(0,1.4,29) #colormap intervals for the difference in persistence probabilities %
cbounds_persall = np.linspace(0,25,26)
colormap_error = plt.cm.rainbow #plt.cm.RdYlBu_r, plt.cm.hot_r, plt.cm.gnuplot2, plt.cm.CMRmap, plt.cm.cubehelix_r, plt.cm.magma_r, plt.cm.jet, plt.cm.afmhot_r colormap used in the maps, must be in this format, is used for plotting the MAE and pattern correlation matrix
colormap_ranking = plt.cm.rainbow
#properties for pcolormesh
snapval = False #parameter for pcolormesh (see below)
#set boxplot options
fliersize = 0.25 #size of the boxplot's fliers
textsize = 7. #tezt size in the plots
flierprops = dict(marker='+', markerfacecolor='black', markersize=6, linestyle='none')
showcaps = False
showfliers = False #False or True
showmeans = False
rotation = 90. #rotation of the model labels in degrees
#properties for the taylor diagram
sigma_lim = 1.55 #1.55, limit of standard deviation ratio used in Taylor plot
corrmethod = 'pearson' #correlation applied for correlation matrix of the error patterns, either pearson, spearman or kendall
groupby = 'agcm' #agcm, ogcm or performance; group by atmosphere or ocean sub-model or by spatial-median performance, from best to worse
complex_thresh = 14
correct_ru = 'no' #correct for the effects of reanalysis uncertainty; if set to "yes", those grid-boxes where interim does not rank 0 (or worse, set by <rank_ru>) within the multi-model ensemble are excluded from the analysis
rank_ru = 0 # if set to 3, South America becomes a "certain" region; performance rank the alternative reanalysis takes in the multi-model ensemble is used as if it was a GCM. If exceeded, the corresponding grid-box is set to nan and is thus excluded from the analyses.
plot_freq = 'yes' #plot an example barplot (GCMs vs. reanalysis) for illustrative purposes
alt_runs = 'no' #loads get_historical_metadata_altruns.py to load alternative runs for a subset of 15 GCMs.
plot_trans_prob = 'no' #plot pcolor figures showing the transition matrix at each grid-box.
edgecolors = 'black' #color of the edges used in these pcolor figures
cmap_probmat = 'hot_r' #colormap used to plot transition probability matrix

#set input parameters for PCA and kmeans clustering (experimental so far, work-in-progress)
exp_var_thresh = 0.95 #explained variance used for selecting number of PCs
km_nr = 3 #number of clusters, 20
km_init = 'k-means++' #init procedure, k-means++ or random
km_n_init = 30 #10
km_max_iter = 50 #300
km_tol = 1e-04
km_random_state = None
stand_for_kmeans = 'no' #space, models or no; set the axis along which the error values are standardized

##EXECUTE###############################################################
#root to input netcdf files
if filesystem == 'extdisk': #for external hard disk
	tarpath = '/media/swen/ext_disk2/datos/lamb_cmip5/results/6h' #path of the source files
	figpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/figs'
	auxpath = home+'/datos/tareas_meteogalicia/lamb_cmip5/pyLamb/aux'
elif filesystem == 'lustre': #for filesystems at IFCA
	tarpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/6h' #path of the source files
	figpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs'
	auxpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb/aux'
else:
    raise Exception('ERROR: unknown entry for <filesystem>!')

dim_wt = 27 #hardcoded because the number of Lamb Weather Types does not change by definition
wt_names = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']

#load GCM metadata
if alt_runs == 'no':
    exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model>
elif alt_runs == 'yes':
    print('INFO: alternative historical runs will be loaded for a subset of 15 GCMs, is they are included in the <model> list specified below.')
    exec(open('get_historical_metadata_altruns.py').read()) #a function assigning alternative runs to the models in <model>
else:
    raise Exception('ERROR: check entry for <alt_runs>!')

#select models to be analysed as a function of the <groupby> parameter, select a given model only once, the specific run for each model (e.g. r1i1p1f1) is assigned by <get_historical_metadata.py> and can be modified there
if groupby == 'agcm':
    print('INFO: The GCMs are grouped according to their AGCM!')
    model = ['jra55','csiro_mk3_6_0','access10','access13','access_cm2','access_esm1_5','hadgem2_es','hadgem2_cc','hadgem3_gc31_mm','kace_1_0_g','fgoals_g2','fgoals_g3','mpi_esm_lr','mpi_esm_mr','mpi_esm_1_2_lr','mpi_esm_1_2_hr','mpi_esm_1_2_ham','awi_esm_1_1_lr','nesm3','cmcc_cm','cmcc_cm2_sr5','cmcc_cm2_hr4','cmcc_esm2','ccsm4','cesm2','noresm1_m','noresm2_lm','noresm2_mm','sam0_unicon','taiesm1','bcc_csm1_1','bcc_csm2_mr','cnrm_cm5','cnrm_cm6_1','cnrm_cm6_1_hr','cnrm_esm2_1','ec_earth','ec_earth3','ec_earth3_veg','ec_earth3_veg_lr','ec_earth3_aerchem','ec_earth3_cc','gfdl_cm3','gfdl_cm4','gfdl_esm2g','gfdl_esm4','kiost_esm','giss_e2_h','giss_e2_r','giss_e2_1_g','ipsl_cm5a_lr','ipsl_cm5a_mr','ipsl_cm6a_lr','miroc5','miroc6','miroc_es2l','miroc_esm','mri_esm1','mri_esm2_0','inm_cm4','inm_cm5','canesm2','iitm_esm'] #'iitm_esm'
    mylabels = ['JRA-55','CSIRO-MK3.6','ACCESS1.0','ACCESS1.3','ACCESS-CM2','ACCESS-ESM1.5','HadGEM2-ES','HadGEM2-CC','HadGEM3-GC31-MM','KACE1.0-G','FGOALS-g2','FGOALS-g3','MPI-ESM-LR','MPI-ESM-MR','MPI-ESM1.2-LR','MPI-ESM1.2-HR','MPI-ESM-1-2-HAM','AWI-ESM-1-1-LR','NESM3','CMCC-CM','CMCC-CM2-SR5','CMCC-CM2-HR4','CMCC-ESM2','CCSM4','CESM2','NorESM1-M','NorESM2-LM','NorESM2-MM','SAM0-UNICON','TaiESM1','BCC-CSM1.1','BCC-CSM2-MR','CNRM-CM5','CNRM-CM6-1','CNRM-CM6-1-HR','CNRM-ESM2-1','EC-Earth2.3','EC-Earth3','EC-Earth3-Veg','EC-Earth3-Veg-LR','EC-Earth3-AerChem','EC-Earth3-CC','GFDL-CM3','GFDL-CM4','GFDL-ESM2G','GFDL-ESM4','KIOST-ESM','GISS-E2-H','GISS-E2-R','GISS-E2.1-G','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM6A-LR','MIROC5','MIROC6','MIROC-ES2L','MIROC-ESM','MRI-ESM1','MRI-ESM2.0','INM-CM4','INM-CM5','CanESM2','IITM-ESM'] #'IITM-ESM'
    # model = ['csiro_mk3_6_0','access10','access13','access_cm2','access_esm1_5','hadgem2_es','hadgem2_cc','hadgem3_gc31_mm','kace_1_0_g','fgoals_g2',]
    # mylabels = ['CSIRO-MK3.6','ACCESS1.0','ACCESS1.3','ACCESS-CM2','ACCESS-ESM1.5','HadGEM2-ES','HadGEM2-CC','HadGEM3-GC31-MM','KACE1.0-G','FGOALS-g2']
elif groupby == 'ogcm':
    print('INFO: The GCMs are grouped according to tpython --versionheir OGCM!')
    model = ['gfdl_cm3','gfdl_cm4','gfdl_esm2g','gfdl_esm4','kiost_esm','csiro_mk3_6_0','access10','access13','access_cm2','access_esm1_5','bcc_csm1_1','bcc_csm2_mr','kace_1_0_g','iitm_esm','hadgem2_es','hadgem2_cc','hadgem3_gc31_mm','mpi_esm_lr','mpi_esm_mr','mpi_esm_1_2_lr','mpi_esm_1_2_hr','mpi_esm_1_2_ham','cmcc_cm','cmcc_cm2_sr5','cmcc_cm2_hr4','cmcc_esm2','cnrm_cm5','cnrm_cm6_1','cnrm_cm6_1_hr','cnrm_esm2_1','ec_earth','ec_earth3','ec_earth3_veg','ec_earth3_veg_lr','ec_earth3_aerchem','ec_earth3_cc','ipsl_cm5a_lr','ipsl_cm5a_mr','ipsl_cm6a_lr','nesm3','awi_esm_1_1_lr','ccsm4','cesm2','sam0_unicon','taiesm1','noresm1_m','noresm2_lm','noresm2_mm','fgoals_g2','fgoals_g3','giss_e2_h','giss_e2_r','giss_e2_1_g','miroc5','miroc6','miroc_esm','miroc_es2l','mri_esm1','mri_esm2_0','inm_cm4','inm_cm5','canesm2']
    mylabels = ['GFDL-CM3','GFDL-ESM2G','GFDL-CM4','GFDL-ESM4','KIOST-ESM','CSIRO-MK3.6','ACCESS1.0','ACCESS1.3','ACCESS-CM2','ACCESS-ESM1.5','BCC-CSM1.1','BCC-CSM2-MR','KACE1.0-G','IITM-ESM','HadGEM2-ES','HadGEM2-CC','HadGEM3-GC31-MM','MPI-ESM-LR','MPI-ESM-MR','MPI-ESM1.2-LR','MPI-ESM1.2-HR','MPI-ESM-1-2-HAM','CMCC-CM','CMCC-CM2-SR5','CMCC-CM2-HR4','CMCC-ESM2','CNRM-CM5','CNRM-CM6-1','CNRM-CM6-1-HR','CNRM-ESM2-1','EC-Earth2.3','EC-Earth3','EC-Earth3-Veg','EC-Earth3-Veg-LR','EC-Earth3-AerChem','EC-Earth3-CC','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM6A-LR','NESM3','AWI-ESM-1-1-LR','CCSM4','CESM2','SAM0-UNICON','TaiESM1','NorESM1-M','NorESM2-LM','NorESM2-MM','FGOALS-g2','FGOALS-g3','GISS-E2-H','GISS-E2-R','GISS-E2.1-G','MIROC5','MIROC6','MIROC-ESM','MIROC-ES2L','MRI-ESM1','MRI-ESM2.0','INM-CM4','INM-CM5','CanESM2']
elif groupby == 'performance':
    print('INFO: The GCMs are grouped according to their median error in ascending order!')
    model = ['csiro_mk3_6_0','access10','access13','access_cm2','access_esm1_5','hadgem2_es','hadgem2_cc','hadgem3_gc31_mm','kace_1_0_g','fgoals_g2','fgoals_g3','mpi_esm_lr','mpi_esm_mr','mpi_esm_1_2_lr','mpi_esm_1_2_hr','mpi_esm_1_2_ham','awi_esm_1_1_lr','nesm3','cmcc_cm','cmcc_cm2_sr5','cmcc_cm2_hr4','cmcc_esm2','ccsm4','cesm2','noresm1_m','noresm2_lm','noresm2_mm','sam0_unicon','taiesm1','bcc_csm1_1','bcc_csm2_mr','cnrm_cm5','cnrm_cm6_1','cnrm_cm6_1_hr','cnrm_esm2_1','ec_earth','ec_earth3','ec_earth3_veg','ec_earth3_veg_lr','ec_earth3_aerchem','ec_earth3_cc','gfdl_cm3','gfdl_cm4','gfdl_esm2g','gfdl_esm4','kiost_esm','giss_e2_h','giss_e2_r','giss_e2_1_g','ipsl_cm5a_lr','ipsl_cm5a_mr','ipsl_cm6a_lr','miroc5','miroc6','miroc_es2l','miroc_esm','mri_esm1','mri_esm2_0','inm_cm4','inm_cm5','canesm2','iitm_esm'] #'iitm_esm'
    mylabels = ['CSIRO-MK3.6','ACCESS1.0','ACCESS1.3','ACCESS-CM2','ACCESS-ESM1.5','HadGEM2-ES','HadGEM2-CC','HadGEM3-GC31-MM','KACE1.0-G','FGOALS-g2','FGOALS-g3','MPI-ESM-LR','MPI-ESM-MR','MPI-ESM1.2-LR','MPI-ESM1.2-HR','MPI-ESM-1-2-HAM','AWI-ESM-1-1-LR','NESM3','CMCC-CM','CMCC-CM2-SR5','CMCC-CM2-HR4','CMCC-ESM2','CCSM4','CESM2','NorESM1-M','NorESM2-LM','NorESM2-MM','SAM0-UNICON','TaiESM1','BCC-CSM1.1','BCC-CSM2-MR','CNRM-CM5','CNRM-CM6-1','CNRM-CM6-1-HR','CNRM-ESM2-1','EC-Earth2.3','EC-Earth3','EC-Earth3-Veg','EC-Earth3-Veg-LR','EC-Earth3-AerChem','EC-Earth3-CC','GFDL-CM3','GFDL-CM4','GFDL-ESM2G','GFDL-ESM4','KIOST-ESM','GISS-E2-H','GISS-E2-R','GISS-E2.1-G','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM6A-LR','MIROC5','MIROC6','MIROC-ES2L','MIROC-ESM','MRI-ESM1','MRI-ESM2.0','INM-CM4','INM-CM5','CanESM2','IITM-ESM'] #'IITM-ESM'
    #model = ['csiro_mk3_6_0','access10','access13','access_cm2','access_esm1_5','hadgem2_es','hadgem2_cc','hadgem3_gc31_mm','kace_1_0_g','fgoals_g2',]
    #mylabels = ['CSIRO-MK3.6','ACCESS1.0','ACCESS1.3','ACCESS-CM2','ACCESS-ESM1.5','HadGEM2-ES','HadGEM2-CC','HadGEM3-GC31-MM','KACE1.0-G','FGOALS-g2']
else:
    raise Exception('ERROR: check entry for <groupby>!')

csvfile = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/'+errortype+'_ref_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'.csv' #path of the csv file containing the median error for each model
yamlfile = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/'+errortype+'_ref_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'.yaml' #path of the csv file containing the median error for each model

print('info: starting analysis for '+region+', '+season_label+', '+experiment+', ' +refdata+', '+errortype+' and complexity threshold '+str(complex_thresh))
print('info: output figures will be saved in '+figfolder+' as '+figformat)
print('info: minimum frequency is '+str(minfreq)+' for '+str(classes_needed)+' LWTs')

#get norm to plot the correlation matrix with pyplot.pcolor(mesh)
cbounds = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
norm_matrix = colors.BoundaryNorm(cbounds, colormap_error.N)
#execfile('get_my_colormap.py') #this auxiliary script returns <my_colormap>, which is currently not used hereafter

if region in ('nh','nh_reduced','eurocordex','cordexna','escena','north_atlantic'):
    hemis = 'nh'
elif region in ('sh','samerica'):
    hemis = 'sh'
else:
    raise Exception('ERROR: check entry for <region>')

#set full path to ERA Interim and JRA-55 LWT catalogues used as reference for verification, remove conditional after recalculating NH catalogues
obs_srcpath_interim = tarpath+'/historical/'+hemis+'/wtseries_interim_historical_r1i1p1_'+hemis+'_'+taryears[0]+'_'+taryears[1]+'.nc'
obs_srcpath_jra55 = tarpath+'/historical/'+hemis+'/wtseries_jra55_historical_r1i1p1_'+hemis+'_'+taryears[0]+'_'+taryears[1]+'.nc'
obs_srcpath_era5 = tarpath+'/historical/'+hemis+'/wtseries_era5_historical_r1i1p1_'+hemis+'_1979_2022.nc'

#initialize lists containing the metadata for each model, which is assigned below in <get_historical_metadata,py>
exec(open('init_runs.py').read())

#assign metadata to each model
for mm in range(len(model)):
    #mrun[mm],doi[mm],atmos[mm],surface[mm],ocean[mm],seaice[mm],aerosols[mm],chemistry[mm],obgc[mm],landice[mm],coupler[mm],complexity[mm],addinfo[mm],family[mm],cmip[mm],rgb[mm],marker[mm],latres_atm[mm],lonres_atm[mm],lev_atm[mm],latres_oc[mm],lonres_oc[mm],lev_oc[mm],ecs[mm],tcr[mm] = get_historical_metadata(model[mm])
    if alt_runs == 'no':
        mrun[mm],complexity[mm],family[mm],cmip[mm],rgb[mm],marker[mm],latres_atm[mm],lonres_atm[mm],lev_atm[mm],latres_oc[mm],lonres_oc[mm],lev_oc[mm],ecs[mm],tcr[mm] = get_historical_metadata(model[mm])
    elif alt_runs == 'yes':
        mrun[mm],complexity[mm],family[mm],cmip[mm],rgb[mm],marker[mm],latres_atm[mm],lonres_atm[mm],lev_atm[mm],latres_oc[mm],lonres_oc[mm],lev_oc[mm],ecs[mm],tcr[mm] = get_historical_metadata_altruns(model[mm])
    else:
        raise Exception('ERROR: check entry for alt_runs!')

taylorprop = np.concatenate((np.expand_dims(np.array(marker),1),np.expand_dims(np.array(rgb),1),np.expand_dims(np.array(rgb),1)),axis=1)
taylorprop[:,2] = 'black'

#get auxiliary variables need for plotting for a specific error type
cbounds_map,errorunit,lowerlim,upperlim = get_error_attrs(errortype)

model_plus_run = [model[ii]+'_'+mrun[ii] for ii in range(len(model))]
model_plus_exp = [model[ii]+'_'+experiment[ii][0] for ii in range(len(model))]
#model_plus_cmip = [model[ii]+'_'+str(int(cmip[ii])) for ii in range(len(model))]
model_plus_cmip = [mylabels[ii]+' '+str(int(cmip[ii])) for ii in range(len(mylabels))]
model_plus_cmip_orig = [mylabels[ii]+' '+str(int(cmip[ii])) for ii in range(len(mylabels))]
    
cbounds_ranking = list(np.arange(0.5,len(model)+1,1))
ticks_ranking = range(1,len(model)+1)

#set a discrete colormap for the ranking maps by first extracting all colors of the continuous colours provided in <colormap_ranking>
cmaplist = [colormap_ranking(i) for i in range(colormap_ranking.N)]
#convert to linearly segmented colormap
colormap_ranking = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, colormap_ranking.N)

##read and load observational data
if refdata == 'interim':
    obs_srcpath = obs_srcpath_interim
elif refdata == 'jra55':
    obs_srcpath = obs_srcpath_jra55
elif refdata == 'era5':
    obs_srcpath = obs_srcpath_era5
else:
    raise Exception('ERROR: unknown entry for <refdata>!')

#get <latlims> and <lonlims>
latlims, lonlims = get_target_coords(region)

obs_dataset = xr.open_dataset(obs_srcpath)

#cut out required time period
dates_obs = pd.DatetimeIndex(obs_dataset.variables['time'].values)
time_ind_obs = (dates_obs.year >= int(taryears[0])) & (dates_obs.year <= int(taryears[1])) & dates_obs.month.isin(season)
obs_dataset = obs_dataset.isel(time=time_ind_obs)
dates_obs = pd.DatetimeIndex(obs_dataset.variables['time'].values)

newlons = obs_dataset.lon.values
newlons[np.where(newlons > 180.)] = newlons[np.where(newlons > 180.)]-360.
obs_dataset.assign_coords(lon=newlons)
ind_roll = np.argmin(np.abs(obs_dataset.lon.values-180))-1
obs_dataset = obs_dataset.roll(lon=ind_roll,roll_coords=True)

#cut out the target region
latind = np.where((obs_dataset.lat >= latlims[0]) & (obs_dataset.lat <= latlims[1]))[0]
lonind = np.where((obs_dataset.lon >= lonlims[0]) & (obs_dataset.lon <= lonlims[1]))[0]
obs_dataset = obs_dataset.isel(lat = latind, lon = lonind)

lats = obs_dataset.variables['lat'][:]
lats_values = lats.values
lons = obs_dataset.variables['lon'][:]
lons_values = lons.values
lons_values[np.where(lons_values>180)] = lons_values[np.where(lons_values>180)]-360
lons_values = lons_values - np.diff(lons_values)[0]/2
halfres = np.gradient(lats_values)[0]/2 #needed below to displace the mesh of the errors for plotting

dates = obs_dataset.variables['time'][:]
obs_wt = obs_dataset.variables['wtseries'][:]
obs_wt = obs_wt.values
dates_pd = pd.to_datetime(dates.values)
periods = dates_pd.to_period(freq = 'M') #is not used in the current version of this script
obs_dataset.close()
timescale = dates_pd.hour[1]-dates_pd.hour[0] #time-scale of the LWT catalogues

#load transition probabilities for reanalysis data or calculate them if they do not exist yet###########################
wtcount = range(1,dim_wt+1)
if errortype in ('TPMS','PERS','PERSall'):
    print('INFO: switching to tpms mode. Observed transition probabilities are loaded or calculated...')
    savedir = tarpath+'/'+experiment[0]+'/transitions/'+region+'/'+season_label.lower()+'/lag'+str(timelag*timescale)+'h'
    if os.path.isdir(savedir) != True:
        os.makedirs(savedir)    
    savepath = savedir+'/trans_prob_'+refdata+'_'+region+'_'+season_label.lower()+'.nc'
    try:
        print('INFO: try loading transition probablities from '+savepath)
        nc = xr.open_dataset(savepath)
        obs_transprob_4d = nc.transition_probabilities.values
        nc.close()
        print('INFO: the try was successful and the file has been loaded....')
    except:
        print('INFO: no file at '+savepath+'. The probabilities are now calculated and saved instead and this will take 12-14 minutes...')
        starttime = time.time()
        obs_transprob_4d = get_transition_probabilities(wtcount,obs_wt,timelag)
        endtime = time.time()
        print('INFO: elapsed time for calculating transition probabilities in '+refdata+' is: '+str((endtime-starttime)/60)+' minutes.')
        print('INFO: the corresponding data is now saved at '+savepath)
        outnc = xr.DataArray(obs_transprob_4d, coords=[lons_values,lats_values,np.array(wtcount),np.array(wtcount)], dims=['lon','lat','LWTi','LWTj'], name='transition_probabilities')
        outnc.to_netcdf(savepath)
        outnc.close()
        del(outnc,savepath)
    if plot_trans_prob == 'yes':
        figpath_trans_prob = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/transition_probabilities/'+refdata
        if os.path.isdir(figpath_trans_prob) != True:
            os.makedirs(figpath_trans_prob)
        print('INFO: The transition probabilities for '+refdata+' are plotted in '+figpath_trans_prob)
        get_fig_transprobmat(obs_transprob_4d,lons_values,lats_values,cmap_probmat,edgecolors,figformat,refdata,figpath_trans_prob,region,dpival,timelag,wt_names)

#get copies for evaluation against hadgem2 models and get rid of December 2005 data (which is lacking for these models) and optionally load or calc transition probabilities
obs_wt_hadgem = np.copy(obs_wt)
dates_hadgem = np.copy(dates)
dates_pd_hadgem = dates_pd.copy(deep=True)
outind = np.where((dates_pd_hadgem.month==12) & (dates_pd_hadgem.year==2005))[0]
obs_wt_hadgem = np.delete(obs_wt_hadgem,outind,axis=0)
dates_hadgem = np.delete(dates_hadgem,outind,axis=0)
del(dates_pd_hadgem)
dates_pd_hadgem = pd.to_datetime(dates_hadgem)

#get various dimensions for error calculations
dim_lon = len(lons)
dim_lat = len(lats)
dim_t = len(dates) #time dimension for obs data, note that time dimension of the models may differ.
dim_t_hadgem = len(dates_hadgem)

if errortype in ('TPMS','PERS','PERSall'):
    #load transition probabilities for reanalysis with December 2005 data removed for use with HadGEM models or calculated them if they do not exist yet###########################
    savepath = tarpath+'/'+experiment[0]+'/transitions/'+region+'/'+season_label.lower()+'/lag'+str(timelag*timescale)+'h/trans_prob_'+refdata+'_hadgem_'+region+'_'+season_label.lower()+'_lag'+str(timelag*timescale)+'h.nc'
    try:
        print('INFO: try loading transition probablities from '+savepath)
        nc = xr.open_dataset(savepath)
        obs_transprob_4d_hadgem = nc.transition_probabilities.values
        nc.close()
        print('INFO: the try was successful and the file has been loaded....')
    except:
        print('INFO: no file at '+savepath+'. The probabilities are now calculated and saved instead and this will take 12-14 minutes...')
        starttime = time.time()
        obs_transprob_4d_hadgem = get_transition_probabilities(wtcount,obs_wt_hadgem,timelag)
        endtime = time.time()
        print('INFO: elapsed time for calculating transition probabilities in '+refdata+' for use w.r.t. HadGEM models is: '+str((endtime-starttime)/60)+' minutes.')
        print('INFO: the corresponding data is now saved at '+savepath)
        outnc = xr.DataArray(obs_transprob_4d_hadgem, coords=[lons_values,lats_values,np.array(wtcount),np.array(wtcount)], dims=['lon','lat','LWTi','LWTj'], name='transition_probabilities')
        outnc.to_netcdf(savepath)
        outnc.close()
        del(outnc,savepath)

#calculate relative observed relative frequencies in any case to rule out those grid boxes where the method is not applicable using <classes_needed> and <minfreq>
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

#load the data and calculate the frequency error or tpms for each GCM
arr_error = np.zeros((dim_lon,dim_lat,len(model)))
arr_mod_freq = np.zeros((dim_wt,dim_lon,dim_lat,len(model)))
for mm in range(len(model)):
    print('Info: start validating '+model[mm]+' and '+mrun[mm])
    #add exception for ERA5 since LWT catalogue ends in 2022
    if cmip[mm] == 1: #1 is a fake number if a reanalysis is to be loaded and verified
        if model[mm] == 'era5':
            print('INFO: LWT catalouge for ERA5 ends in 2022.')
            mod_srcpath = tarpath +'/'+ experiment[mm] + '/'+hemis+'/wtseries_' + model[mm] + '_' + experiment[mm] +'_'+ mrun[mm] +'_' +hemis+'_'+taryears[0] +'_2022.nc'
        else:
            mod_srcpath = tarpath +'/'+ experiment[mm] + '/'+hemis+'/wtseries_' + model[mm] + '_' + experiment[mm] +'_'+ mrun[mm] +'_' +hemis+'_'+taryears[0] +'_'+ taryears[1]+'.nc'
    elif cmip[mm] == 6: #for CMIP6 models starting in 1850 and ending in 2014
        if (model[mm] == 'ec_earth3_veg') & (mrun[mm] in ('r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1','r11i1p1f1')) or (model[mm] == 'mpi_esm_1_2_hr') & (mrun[mm] in ('r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1')):
            mod_srcpath = tarpath +'/'+ experiment[mm] + '/'+hemis+'/wtseries_' + model[mm] + '_' + experiment[mm] +'_'+ mrun[mm] +'_' +hemis+'_'+taryears_cmip6_long[0] +'_'+ taryears_cmip6_long[1]+'.nc'
            print('INFO: The LWT catalogue for '+model[mm]+', '+experiment[mm]+', '+mrun[mm]+' is available from '+str(taryears_cmip6_long[0])+' to '+str(taryears_cmip6_long[1]))
        else: #for CMIP6 models starting in 1979 and ending in 2014
            mod_srcpath = tarpath +'/'+ experiment[mm] + '/'+hemis+'/wtseries_' + model[mm] + '_' + experiment[mm] +'_'+ mrun[mm] +'_' +hemis+'_'+taryears_cmip6[0] +'_'+ taryears_cmip6[1]+'.nc'
            print('INFO: The LWT catalogue for '+model[mm]+', '+experiment[mm]+', '+mrun[mm]+' is available from '+str(taryears_cmip6[0])+' to '+str(taryears_cmip6[1]))
    elif cmip[mm] == 5: #for CMIP5 models starting in 1979 and ending in 2005
        mod_srcpath = tarpath +'/'+ experiment[mm] + '/'+hemis+'/wtseries_' + model[mm] + '_' + experiment[mm] +'_'+ mrun[mm] +'_' +hemis+'_'+taryears[0] +'_'+ taryears[1]+'.nc'
        print('INFO: The LWT catalogue for '+model[mm]+', '+experiment[mm]+', '+mrun[mm]+' is available from '+str(taryears[0])+' to '+str(taryears[1]))
    else:
        raise Exception('ERROR: chech <taryears>, <taryears_cmip6>, <taryears_cmip6_long> or <cmip> for <model[mm]> !')

    #load the model or reanalysis data to be verified
    mod_dataset = xr.open_dataset(mod_srcpath)
    print('The first time instance for '+model[mm]+' and '+mrun[mm]+' is: '+str(mod_dataset.time[0].values))    
    
    #cut out required time period
    dates_mod = pd.DatetimeIndex(mod_dataset.variables['time'].values)
    time_ind_mod = (dates_mod.year >= int(taryears[0])) & (dates_mod.year <= int(taryears[1])) & dates_mod.month.isin(season)
    mod_dataset = mod_dataset.isel(time=time_ind_mod)
    dates_mod = pd.DatetimeIndex(mod_dataset.variables['time'].values)
    
    #roll the longitudes
    newlons = mod_dataset.lon.values
    newlons[np.where(newlons > 180.)] = newlons[np.where(newlons > 180.)]-360.
    mod_dataset.assign_coords(lon=newlons)
    ind_roll = np.argmin(np.abs(mod_dataset.lon.values-180))-1
    mod_dataset = mod_dataset.roll(lon=ind_roll,roll_coords=True)    

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
    
    if errortype in ('TPMS','PERS','PERSall'):
        #load transition probabilities for GCM data or calculate them if they do not exist yet ###########################
        print('INFO: getting transition probablities for '+model_plus_run[mm])
        savepath = tarpath+'/'+experiment[mm]+'/transitions/'+region+'/'+season_label.lower()+'/lag'+str(timelag*timescale)+'h/trans_prob_'+model_plus_run[mm]+'_'+region+'_'+season_label.lower()+'_lag'+str(timelag*timescale)+'h.nc'
        try:
            print('INFO: try loadinng transition probablities from '+savepath)
            nc = xr.open_dataset(savepath)
            mod_transprob_4d = nc.transition_probabilities.values
            nc.close()
            print('INFO: the try was successful and the file has been loaded....')
        except:
            print('INFO: no file at '+savepath+'. The probabilities are now calculated and saved instead and this will take 12-14 minutes...')
            starttime = time.time()
            mod_transprob_4d = get_transition_probabilities(wtcount,mod_wt,timelag)
            endtime = time.time()
            print('INFO: elapsed time for calculating transition probabilities in '+model_plus_run[mm]+' is: '+str((endtime-starttime)/60)+' minutes.')
            print('INFO: the corresponding data is now saved at '+savepath)
            outnc = xr.DataArray(mod_transprob_4d, coords=[lons_values,lats_values,np.array(wtcount),np.array(wtcount)], dims=['lon','lat','LWTi','LWTj'], name='transition_probabilities')
            outnc.to_netcdf(savepath)
            outnc.close()
            del(outnc,savepath)
        if plot_trans_prob == 'yes':
            figpath_trans_prob = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/transition_probabilities/'+model_plus_run[mm]
            if os.path.isdir(figpath_trans_prob) != True:
                os.makedirs(figpath_trans_prob)
            print('INFO: The modelled transition probabilities are plotted in '+figpath_trans_prob)
            get_fig_transprobmat(mod_transprob_4d,lons_values,lats_values,cmap_probmat,edgecolors,figformat,model_plus_run[mm],figpath_trans_prob,region,dpival,timelag,wt_names)
        if model[mm] in ('hadgem2_es','hadgem2_cc'):
            if errortype in ('TPMS','PERS'):
                probdiff = mod_transprob_4d - obs_transprob_4d_hadgem
            elif errortype == 'PERSall':
                probdiff = np.sum(np.diagonal(mod_transprob_4d,axis1=3,axis2=2),axis=2) - np.sum(np.diagonal(obs_transprob_4d_hadgem,axis1=3,axis2=2),axis=2)
            else:
                raise Exception('ERROR: Check entry for <errortype>!')
        else:
            if errortype in ('TPMS','PERS'):
                probdiff = mod_transprob_4d - obs_transprob_4d
            elif errortype == 'PERSall':
                probdiff = np.sum(np.diagonal(mod_transprob_4d,axis1=3,axis2=2),axis=2) - np.sum(np.diagonal(obs_transprob_4d,axis1=3,axis2=2),axis=2)
            else:
                raise Exception('ERROR: Check entry for <errortype>!')            
            
        #reshape to 3d array and calc error
        if errortype == 'TPMS':
            probdiff = probdiff.reshape(probdiff.shape[0],probdiff.shape[1],probdiff.shape[2]*probdiff.shape[2])
            error = np.mean(np.abs(probdiff)*100,axis=2)
        #get diagonal in case persistence probablitiy is evaluated, this already leads to the desired shape
        elif errortype == 'PERS':
            probdiff = np.diagonal(probdiff,axis1=3,axis2=2)
            error = np.mean(np.abs(probdiff)*100,axis=2)
        # resphaping is not necessary for PERSall
        elif errortype == 'PERSall':
            error = np.abs(probdiff)*100
        else:
            raise Exception('ERROR: check entry for <errortype>!')

    elif errortype in ('MAE','KL'):
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
            raise Exception('ERROR: check entry for <errortype>!')
    else:
        raise Exception('ERROR: check entry for <errortype>!')
    arr_error[:,:,mm] = error

##find performance rank of each model at each grid box
id_error = np.zeros(arr_error.shape)
for kk in list(range(arr_error.shape[0])):
        for zz in list(range(arr_error.shape[1])):
            #id_error[kk,zz,:] = np.argsort(arr_error[kk,zz,:])
            temp = np.argsort(arr_error[kk,zz,:])
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(arr_error[kk,zz,:]))
            #set ranks to nan if nans are present in the error array for this grid-box
            # if any(np.isnan(arr_error[kk,zz,:])):
                # ranks[temp] = np.arange(len(arr_error[kk,zz,:]))/np.nan
            # else:
                # ranks[temp] = np.arange(len(arr_error[kk,zz,:]))            
            id_error[kk,zz,:] = ranks
            
#id_error = np.argsort(arr_error,axis=2).astype('float')+1

##find the best performing model at each grid box
id_best = np.argmin(arr_error,axis=2)+1. #best rank is 1

#take into account reanalysis uncertainty
if correct_ru == 'yes':
    if (region == 'nh' and errortype in ('MAE','TPMS','PERS','PERSall','KL') and classes_needed == 27 and minfreq == 0.001) or (region == 'sh' and errortype in ('MAE','TPMS','PERS','PERSall','KL') and classes_needed == 20 and minfreq == 0.001):
        print('INFO: model error and ranks where ERA-Interim vs. JRA-55 does not perform best first will be set to nan!')
        rean_file = auxpath+'/rank_'+errortype+'_interim_jra55_'+region+'_1979_2005.nc'
        #rean_file = auxpath+'/rank_MAE_interim_jra55_'+region+'_1979_2005.nc'
        
        #use these commmands to create and save <corrnc>
        #rank_interim_vs_jra55_in_ensemble = id_error[:,:,0]
        #ds = xr.Dataset(data_vars=dict(rank_interim_vs_jra55_in_ensemble=(["x", "y"], rank_interim_vs_jra55_in_ensemble),),coords=dict(lon=(["x"], range(len(lons))),lat=(["y"], range(len(lats))),),attrs=dict(description="Performance rank of ERA-Interim w.r.t. JRA-55 in multi-model ensemble"),)
        #ds.to_netcdf(rean_file)
        #ds.close()
        corrnc = xr.open_dataset(rean_file) #contains n replicas of the matrix along the third axis, with n = number of models
        rank_interim_vs_jra55_in_ensemble = np.repeat(np.expand_dims(corrnc.rank_interim_vs_jra55_in_ensemble.values,axis=2),len(model),axis=2)
        rean_nanmask = np.where(rank_interim_vs_jra55_in_ensemble > rank_ru)
        arr_error[rean_nanmask] = np.nan
        id_error[rean_nanmask] = np.nan
    else:
        raise Exception('ERROR: <correct_ru> is not defined for '+region+', '+errortype+', '+str(classes_needed)+', '+str(minfreq))
else:
    print('INFO: reanalysis uncertainty is not taken into account. If you like to filter out those grid-boxes where this kind of uncertainty is enhanced, set <correct_ru> to yes and re-run this script.')

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
arr_error_2d_gcms = np.copy(arr_error_2d) #retain initial matrix including individual model results only. 
median_error = np.nanmedian(arr_error_2d,axis=0)
mean_error = np.nanmean(arr_error_2d,axis=0)

#group GCM errors and all associated objects by performance
if groupby == 'performance':
    exec(open('sortby_performance.py').read())

#get all Pearson (rho), Spearman (spear) and Kendall (tau) correlation coefficients between median error and the various resolutions
exec(open('get_corr.py').read())

#get errors for cmip5 and 6 and append arr_error_2d
#get indices
cmip5_ind = np.where(cmip == 5)[0]
cmip6_ind = np.where(cmip == 6)[0]

#sum the integer codes to obtain an ascending complexity estimate
complex_sum = np.zeros(len(complexity)) #initialize array and fill it thereafter
for ii in range(len(complexity)):
    complex_step = complexity[ii]
    complex_sum[ii] = np.sum([int(complex_step[jj]) for jj in range(len(complex_step))])

gcm_ind_5 = np.where((complex_sum < complex_thresh) & (np.array(cmip) == 5))[0]
gcm_ind_6 = np.where((complex_sum < complex_thresh) & (np.array(cmip) == 6))[0]
esm_ind_5 = np.where((complex_sum >= complex_thresh) & (np.array(cmip) == 5))[0]
esm_ind_6 = np.where((complex_sum >= complex_thresh) & (np.array(cmip) == 6))[0]

#get flattened arrays
arr_error_cmip5 = arr_error_2d[:,cmip5_ind]
arr_error_cmip6 = arr_error_2d[:,cmip6_ind]
arr_error_gcm_5 = arr_error_2d[:,gcm_ind_5].flatten()
arr_error_gcm_6 = arr_error_2d[:,gcm_ind_6].flatten()
arr_error_esm_5 = arr_error_2d[:,esm_ind_5].flatten()
arr_error_esm_6 = arr_error_2d[:,esm_ind_6].flatten()

#get max. length of the flattened arrays
maxlen = np.nanmax((len(arr_error_gcm_5),len(arr_error_esm_5),len(arr_error_gcm_6),len(arr_error_esm_6)))
#get length differences for each array
difflen_mat = maxlen - arr_error_2d.shape[0]
difflen_gcm_5 = maxlen - len(arr_error_gcm_5)
difflen_esm_5 = maxlen - len(arr_error_esm_5)
difflen_gcm_6 = maxlen - len(arr_error_gcm_6)
difflen_esm_6 = maxlen - len(arr_error_esm_6)
#expand the arrays
#begin with the matrix
nanmat = np.empty((difflen_mat,arr_error_2d.shape[1]))
nanmat[:] = np.nan
arr_error_2d = np.concatenate((arr_error_2d,nanmat),axis=0) #concatenate rows with nans

#then lengthen flattened arrays if necessary
nanvec_gcm_5 = np.repeat(np.nan,difflen_gcm_5)
nanvec_esm_5 = np.repeat(np.nan,difflen_esm_5)
nanvec_gcm_6 = np.repeat(np.nan,difflen_gcm_6)
nanvec_esm_6 = np.repeat(np.nan,difflen_esm_6)
#concatenate flattened arrays
arr_error_gcm_5 = np.concatenate((arr_error_gcm_5,nanvec_gcm_5),axis=0)
arr_error_esm_5 = np.concatenate((arr_error_esm_5,nanvec_esm_5),axis=0)
arr_error_gcm_6 = np.concatenate((arr_error_gcm_6,nanvec_gcm_6),axis=0)
arr_error_esm_6 = np.concatenate((arr_error_esm_6,nanvec_esm_6),axis=0)

#finally concatanate the extended matrix with the 4 extended flattened arrays
arr_error_2d = np.concatenate((arr_error_2d,np.expand_dims(arr_error_gcm_5,1),np.expand_dims(arr_error_esm_5,1),np.expand_dims(arr_error_gcm_6,1),np.expand_dims(arr_error_esm_6,1)),axis=1) #concatente columns with cmip5 and 6 summary results
#add addiotional columns to <model_plus_cmip> or equivalent and also expand the rgb list, then define color_dict dictionary
model_plus_cmip = model_plus_cmip+['less complex 5','more complex 5','less complex 6','more complex 6']
rgb = rgb+['#98ff98','#98ff98','#98ff98','#98ff98']

#get rid of columns only containing nans (because their are either no ESMs or AOGCMs or CMIP5 or CMIP6 models), and create colour dictionary
getind = np.where(np.sum(np.isnan(arr_error_2d),axis=0) != maxlen)[0]
arr_error_2d = arr_error_2d[:,getind]
rgb = list(np.array(rgb)[getind])
model_plus_cmip =  list(np.array(model_plus_cmip)[getind])
color_dict = dict(zip(model_plus_cmip, rgb))

##boxplot with seaborn
fig = sns.boxplot(data=arr_error_2d,orient='v',fliersize=fliersize,palette=rgb)
plt.subplots_adjust(bottom=0.215) #see https://www.python-graph-gallery.com/192-about-matplotlib-margins
fig.set_xticklabels(model_plus_cmip,rotation=rotation,size=textsize) #model_plus_exp
fig.set_ylim(lowerlim,upperlim)
fig.set_ylabel(errortype+' of relatative LWT frequencies ('+errorunit+')', size=9.)

savedir_boxplot = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()
if os.path.isdir(savedir_boxplot) != True:
    os.makedirs(savedir_boxplot)
savepath = savedir_boxplot+'/boxplot_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savepath, dpi=dpival)
plt.close('all')

savedir_error_map = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/maps/'
if os.path.isdir(savedir_error_map) != True:
    os.makedirs(savedir_error_map)
norm = mpl.colors.BoundaryNorm(cbounds_ranking, colormap_ranking.N)
#plot errors and ranking for each model
for mm in range(len(model)):    
    #error map
    cbar_error = errortype.upper()+' of relative LWT frequencies ('+errorunit+')'
    title_error = errortype+' '+model[mm]+' '+mrun[mm]+' w.r.t. '+refdata.upper()+' '+str(taryears[0])+'-'+str(taryears[1])
    savename_error = savedir_error_map+'/'+errortype+'_'+model[mm]+'_'+mrun[mm]+'_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
    draw_error_map('error',region,lats_values,lons_values,np.transpose(arr_error[:,:,mm]),colormap_error,halfres,cbounds_map,snapval,savename_error,title_error,cbar_error,figformat,textsize,dpival,norm=None,ticks_cbar=None)
    
    ##rank map
    cbar_rank = 'Rank of '+errortype+' ('+errorunit+')'
    title_rank = 'Rank of '+errortype+' '+model[mm]+' '+mrun[mm]+' w.r.t. '+refdata.upper()+' '+str(taryears[0])+'-'+str(taryears[1])
    savename_rank = savedir_error_map+'/rank_'+model[mm]+'_'+mrun[mm]+'_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
    draw_error_map('rank',region,lats_values,lons_values,np.transpose(id_error[:,:,mm]),colormap_ranking,halfres,cbounds_map,snapval,savename_rank,title_rank,cbar_rank,figformat,textsize,dpival,norm=norm,ticks_cbar=ticks_ranking)

##plot the Taylor diagrams, first get the LWT per model, then flatten data for each model and finally calc statistics and draw plot for this LWT
savedir_taylor = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/taylor'
if os.path.isdir(savedir_taylor) != True:
    os.makedirs(savedir_taylor)
stat_all = np.zeros((dim_wt,arr_mod_freq.shape[3],6))
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
    #ax.legend(model_plus_cmip[0:-4])
    savepath = savedir_taylor+'/LWT_'+str(ww+1)+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
    plt.savefig(savepath, dpi=dpival)
    plt.close('all')
    stat_all[ww,:,:] = stat

#save results in netcdf (all errors) and csv (median errors only) format
main_results = pd.DataFrame(data=np.round(median_error,4), index=model_plus_run, columns=[errortype])
main_results['cmip'] = cmip.astype('int')
#family_bin = np.array(family)
#family_bin[np.where(np.array(family) == 'gcm')] = 0
#family_bin[np.where(np.array(family) == 'esm')] = 1
#main_results['esm'] = list(family_bin)
main_results['complexity'] = list(complexity)
main_results['reference'] = list(np.tile(refdata,len(model)))
main_results.to_csv(csvfile,header=[errortype,'cmip','complexity','reference'])

#save grid-box errors in 1d array per model for further processing with <make_scatterplot.py>
outnc = xr.DataArray(arr_error_2d_gcms, coords=[range(arr_error_2d_gcms.shape[0]),model], dims=['grid-box-errors', 'gcm'], name=errortype)
outnc.attrs['description'] = 'grid-box-scale '+errortype+' for '+refdata+', '+region+', '+str(classes_needed)+', '+str(minfreq)
outnc.attrs['unit'] = errorunit
outnc.attrs['reference_reanalysis'] = refdata
outnc.attrs['documentation'] = 'doi: 10.5194/gmd-15-1375-2022'
outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
savepath_outnc = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/gridbox_errors_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.nc'
print('INFO: the grib-box-scale circulation errors are saved at '+savepath_outnc)
outnc.to_netcdf(savepath_outnc)
outnc.close()
del(outnc,savepath_outnc)

#save grid-box errors in lon x lat x gcm array
outnc = xr.DataArray(arr_error, coords=[lons,lats,model_plus_run], dims=['lon','lat','gcm'], name=errortype)
outnc.attrs['description'] = 'grid-box-scale '+errortype+' for '+refdata+', '+region+', '+str(classes_needed)+', '+str(minfreq)+' in a lon x lat x gcm array'
outnc.attrs['unit'] = errorunit
outnc.attrs['reference_reanalysis'] = refdata
outnc.attrs['documentation'] = '1. GRL manuscript 2022GL101446 and 2. doi: 10.5194/gmd-15-1375-2022'
outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
savepath_outnc = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/lon_x_lat_x_gcm_errors_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.nc'
print('INFO: the grib-box-scale circulation errors in lon x lat x gcm format are saved at '+savepath_outnc)
outnc.to_netcdf(savepath_outnc)
outnc.close()
del(outnc,savepath_outnc)

#correlate with ecs and tcr
tcr = np.array(tcr)
ecs = np.array(ecs)

gcm5ind = (np.where(family=='gcm')) and (np.where(cmip==5))[0]
gcm6ind = (np.where(family=='gcm')) and (np.where(cmip==6))[0]
esm5ind = (np.where(family=='esm')) and (np.where(cmip==5))[0]
esm6ind = (np.where(family=='esm')) and (np.where(cmip==6))[0]

#plot median mae vs tcr and ecs and get correlation coefficients
mycolors=np.array(['blue']*len(model))
mycolors[np.where(cmip==6)]='red'
mycolors = list(mycolors)
markers=np.array(['o']*len(model))
markers[np.where(np.array(family)=='esm')]='d'
markers = list(markers)

#calculate correlation between LWT error and TCR or ECS
r_tcr = stats.spearmanr(median_error,tcr,nan_policy='omit')
r_ecs = stats.spearmanr(median_error,ecs,nan_policy='omit')

#remove nans (present in tcr and ecs) and make a scatterplot
nanind = np.where(np.isnan(tcr))[0]
median_error_plot = np.delete(median_error,nanind)
tcr_plot = np.delete(tcr,nanind)
mycolors_plot = list(np.delete(np.array(mycolors),nanind))
markers_plot = list(np.delete(np.array(markers),nanind))
model_plot = list(np.delete(np.array(model),nanind))

fig = plt.figure()
ax = fig.gca()
for ii in list(range(len(mycolors_plot))):
    ax.plot(median_error_plot[ii],tcr_plot[ii],color=mycolors_plot[ii],marker=markers_plot[ii])
    #ax.text(median_error[ii],tcr_plot[ii],model_plot[ii],style='italic')
ax.text(median_error_plot.max(),tcr_plot.max(),'r = '+str(round(r_tcr.correlation,2)),style='italic')
ax.set_aspect(1./ax.get_data_ratio())
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth(1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Median '+errortype,size=12)
plt.ylabel('TCR at time of CO2 doubling (K)',size=12)
#forceAspect(ax,aspect=1)
#plt.legend()
savename_error_tcr = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/'+errortype+'_vs_tcr_2xCO2_'+refdata+'_'+groupby+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savename_error_tcr,dpi=dpival)
plt.close('all')

#generate a scatterplot for the relationship between model complexity and performance and calculate Spearman and Pearson correlation coefficients
#get sums of the complexity codes
complex_sum = np.array([np.sum(np.array(list(complexity[ii])).astype('int')) for ii in range(len(complexity))])
spear_complex_err = stats.spearmanr(complex_sum,median_error,nan_policy='omit')
rho_complex_err = stats.pearsonr(complex_sum,median_error)
fig = plt.figure()
ax = fig.gca()
for ii in list(range(len(median_error))):
    ax.plot(complex_sum[ii],median_error[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
ax.set_aspect(1./ax.get_data_ratio())
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth(1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Model complexity',size=12)
plt.ylabel('Median '+errortype,size=12)
#forceAspect(ax,aspect=1)
#plt.legend()
plt.savefig(figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/complexity_vs_'+errortype+'_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat)
plt.close('all')

#generate a scatterplot for the relationship between horizontal or 3D resolution of the AGCM and performance, note that the mesh sizes are calculated in get_corr.py
fig = plt.figure()
ax = fig.gca()
for ii in list(range(len(median_error))):
    #ax.plot(meshsize[ii],median_error[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
    #ax.plot(np.log(meshsize[ii]),median_error[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
    ax.plot(res_atm[ii],median_error[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
    #ax.plot(res_oc[ii],median_error[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth(1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.xlabel('Mesh size of the AGCM',size=12)
plt.xlabel('3D mesh size of the AGCM',size=12)
plt.ylabel('Median '+errortype+' of LWT frequencies (%)',size=12)
#plt.xlim([0,150000]) #for 2d meshsize
plt.xlim([0,1.5*10**7]) #for 3d meshsize
ax.set_aspect(1./ax.get_data_ratio())
#forceAspect(ax,aspect=1)
#plt.legend()
plt.savefig(figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/meshsize_agcm_vs_'+errortype+'_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat)
plt.close('all')

#generate and save yaml file to be used in the EURO-CORDEX github entry, as suggested by Jesús Fernández, CSIC
yaml_cont = {'key':'Brands21 Lamb '+region, 'doi':'10.5194/gmd-2020-418', 'metric':{'name':errortype.upper(),'long_name':'mean absolute error in relative frequenies of simulated vs. reanalysis Lamb Weather Types','units':errorunit,'variables':'psl','comment':'see doi:10.1029/2019GL086695', \
    'best':'0', 'worst':'100'}, 'type':'performance', 'spatial_scope':'EUR', 'temporal_scopre':'Annual', 'period':{'reference':'1979-2004'}, 'plausible values':{'min':str(np.min(median_error)),'max':str(np.max(median_error)),'source':'eurocordex_gcm_selection_team', \
    'comment':'These are the Brands 2021 results for the EURO-CORDEX domain only, as requested during the 2021-05-28 meeting on GCM selection.'}, 'data_source':'author', 'data':{model_plus_run[0]:str(median_error[0])}}
print('saving main results in yaml format...')
# with open(yamlfile, 'w') as outfile:
    # yaml.dump(yaml_cont, outfile, default_flow_style=True)
# outfile.close()

gitfile = open(yamlfile, "w")
yaml.dump(yaml_cont, gitfile, default_flow_style=False)
gitfile.close()

#get correlation matrix for the MAE error patterns of the models,  Pandas DataFrame is used to treat NaNs
#bring data to pandas format
arr_error_2_corrmat = pd.DataFrame(np.reshape(arr_error,[arr_error.shape[0]*arr_error.shape[1],arr_error.shape[2]]))
arr_error_2_corrmat_cmip5 = pd.DataFrame(arr_error_cmip5)
arr_error_2_corrmat_cmip6 = pd.DataFrame(arr_error_cmip6)

#compute correlation matrices and plot
corrmat = arr_error_2_corrmat.corr(method=corrmethod)
corrmat_cmip5 = arr_error_2_corrmat_cmip5.corr(method=corrmethod)
corrmat_cmip6 = arr_error_2_corrmat_cmip6.corr(method=corrmethod)
corrmat_nandiag = corrmat.copy()
corrmat_nandiag[corrmat==1.] = np.nan #set diagonal to nan for calculating mean and median in the next two lines

#calc mean and median error pattern correlation coefficients to be printed in along the axes of the correlation matrix plot
mean_corr_raw = np.nanmean(corrmat_nandiag.values,axis=1) # mean r, raw = not rounded
mean_corr = np.round(mean_corr_raw*100).astype('int') # mean r x 100 and rounded
median_corr = np.round(np.nanmedian(corrmat_nandiag.values,axis=1)*100).astype('int') # median r x 100 and rounded

#calc weights (1 - r)
weight_1 = 1. - mean_corr_raw

#get alternative weights with square of mean correlation coefficients
weight_2 = np.empty_like(weight_1)
negind = np.where(mean_corr_raw < 0.)
posind = np.where(mean_corr_raw >= 0.)
weight_neg = 1 + mean_corr_raw[negind]**2
weight_pos = 1 - mean_corr_raw[posind]**2
weight_2[negind] = weight_neg
weight_2[posind] = weight_pos

#get rounded weights
#mean_weight_alt = np.round(1. - np.nanmean(corrmat_nandiag.values,axis=1),2)
#median_weight_alt = np.round(1. - np.nanmedian(corrmat_nandiag.values,axis=1),2)

#assign mean correlations and weights to models using strings
model_plus_cmip_plus_corr = [model_plus_cmip_orig[ii]+' '+str(mean_corr[ii]) for ii in range(len(mylabels))]
model_plus_cmip_plus_weight_1 = [model_plus_cmip_orig[ii]+' '+str(np.round(weight_1[ii],2)) for ii in range(len(mylabels))]
model_plus_cmip_plus_weight_2 = [model_plus_cmip_orig[ii]+' '+str(np.round(weight_2[ii],2)) for ii in range(len(mylabels))]

#save both types of weights in netcdf format
outnc = xr.DataArray(np.stack((weight_1,weight_2)), coords=[list(('method1','method2')),model_plus_run], dims=['method','gcm'], name='weights')
outnc.attrs['description'] = 'Weighting factors describing the degree of GCM independence as described in GRL manuscript 2022GL101446; errortype: '+errortype+', reference reanalysis: '+refdata+', region: '+region+', number of required Lamb classes:'+str(classes_needed)+', miniumum frequency:'+str(minfreq)
outnc.attrs['unit'] = 'continuous from 0 to 2, proportional to the degree of independence'
outnc.attrs['reference_reanalysis'] = refdata
outnc.attrs['method1'] = '1 - r, where r is the average of '+str(len(model)-1)+' correlation coefficients obtained from correlating the spatial error pattern of the indicated GCM with the error patterns of n='+str(len(model)-1)+' other GCMs.'
outnc.attrs['method2'] = '1 - r^2 for r >= 0 and 1 + r^2 for r < 0, where r is the average of '+str(len(model)-1)+' correlation coefficients obtained from correlating the spatial error pattern of the indicated GCM with the error patterns of n='+str(len(model)-1)+' other GCMs.'
outnc.attrs['documentation'] = '1. GRL manuscript 2022GL101446 and 2. doi: 10.5194/gmd-15-1375-2022'
outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
outnc.method.attrs['method1'] = '1 - r, where r is the average of '+str(len(model)-1)+' correlation coefficients obtained from correlating the spatial error pattern of the indicated GCM with the error patterns of n='+str(len(model)-1)+' other GCMs.'
outnc.method.attrs['method2'] = '1 - r^2 for r >= 0 and 1 + r^2 for r < 0, where r is the average of '+str(len(model)-1)+' correlation coefficients obtained from correlating the spatial error pattern of the indicated GCM with the error patterns of n='+str(len(model)-1)+' other GCMs.'
outnc.gcm.attrs['description'] = 'https://github.com/SwenBrands/gcm-metadata-for-cmip/blob/main/get_historical_metadata.py'
savepath_outnc = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/gcm_weights_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.nc'
print('INFO: the weighting factors per GCM are saved at '+savepath_outnc)
outnc.to_netcdf(savepath_outnc)
outnc.close()
del(outnc,savepath_outnc)

#plot the correlation matrix
fig = plt.pcolor(corrmat,cmap=colormap_error,norm=norm_matrix)
plt.title('Pattern correlation for the pointwise '+errortype+' over '+region.upper()+' domain '+season_label,size=textsize)
plt.margins(x=0, y=0)
plt.xticks(np.array(range(0,len(model)))+0.5)
plt.subplots_adjust(bottom=0.18) #see https://www.python-graph-gallery.com/192-about-matplotlib-margins
fig.axes.set_xticklabels(model_plus_cmip_plus_corr,rotation=rotation,size=textsize-2)
plt.yticks(np.array(range(0,len(model)))+0.5)
fig.axes.set_yticklabels(model_plus_cmip_plus_corr,rotation=0,size=textsize-2)
#cbar = plt.colorbar(label=corrmethod[0].upper()+corrmethod[1:].lower()+' correlation coefficient',shrink=0.75, ticks=cbounds[::2])
cbar = plt.colorbar(shrink=0.75, ticks=cbounds[::2], extend='min', drawedges=True)
cbar.set_label(corrmethod[0].upper()+corrmethod[1:].lower()+' correlation coefficient',size=textsize)
cbar.ax.tick_params(labelsize=textsize-1,labelright=textsize-2)
fig.axes.axes.set_aspect(1./fig.axes.axes.get_data_ratio()) #set equal axis lengths
savepath = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/corrmat_'+corrmethod+'_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savepath, dpi=dpival)
plt.close('all')

#plot boxplot of the flattened correlation matrix without unity values, do this for all models (flatrho) and separately for CMIP5 and 6 versions (flatrho_cmip5 6), then join all three arrays in flatrho_3
flatrho = np.ndarray.flatten(corrmat.values) #flatten the matrix
flatrho = np.unique(np.delete(flatrho,np.where(flatrho==1.))) #remove unity values along the former diagonal
flatrho_cmip5 = np.ndarray.flatten(corrmat_cmip5.values)
flatrho_cmip5 = np.unique(np.delete(flatrho_cmip5,np.where(flatrho_cmip5==1.)))
flatrho_cmip6 = np.ndarray.flatten(corrmat_cmip6.values)
flatrho_cmip6 = np.unique(np.delete(flatrho_cmip6,np.where(flatrho_cmip6==1.)))
maxlen = len(flatrho)
nan_cmip5 = np.repeat(np.nan,maxlen-len(flatrho_cmip5))
nan_cmip6 = np.repeat(np.nan,maxlen-len(flatrho_cmip6))
flatrho_cmip5 = np.concatenate((flatrho_cmip5,nan_cmip5),axis=0)
flatrho_cmip6 = np.concatenate((flatrho_cmip6,nan_cmip6),axis=0)
flatrho_3 = np.concatenate((np.expand_dims(flatrho,1),np.expand_dims(flatrho_cmip5,1),np.expand_dims(flatrho_cmip6,1)),axis=1)

fig = plt.figure()
flierprops = dict(marker='x', markerfacecolor='black', markersize=12,linestyle='none')
sns.boxplot(data=flatrho,orient='v',width=0.1,color='white')
savepath_boxplot = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/boxplot_patterncorr_'+corrmethod+'_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savepath_boxplot, dpi=dpival)
plt.close('all')

#save correlation matrix in netCDF format
print('INFO: The overall average '+corrmethod+' correlation coefficient is: '+str(np.mean(corrmat.values)))
outnc = xr.DataArray(corrmat, coords=[model_plus_run, model_plus_run], dims=['gcmrun', 'gcmrun'], name='rho')
outnc.attrs['description'] = 'rho is the spatial correlation of the mean absolute error in the relative frequencies of the 27 Lamb Weather Types w.r.t to '+refdata+' over the '+region+' domain, for a given GCM pair.'
outnc.attrs['unit'] = corrmethod[0].upper()+corrmethod[1:].lower()+' correlation coefficient'
outnc.attrs['reference'] = 'doi: 10.5194/gmd-15-1375-2022'
outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
savepath_outnc = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/corrmat_'+corrmethod+'_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.nc'
print('INFO: the correlation matrix describing inter model dependencies is saved at '+savepath_outnc)
outnc.to_netcdf(savepath_outnc)
outnc.close()

#perform k-means clustering, see https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters, https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
#and https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#bring error array to the right format
arr_error_2_kmeans = arr_error.reshape(arr_error.shape[0]*arr_error.shape[1],arr_error.shape[2]).transpose()
#remove nans prior to pca and kmeans clustering
ind_nonan = ~np.isnan(arr_error_2_kmeans).any(axis=0) #for removing nans see https://note.nkmk.me/en/python-numpy-nan-remove/
arr_error_2_kmeans_nonans = arr_error_2_kmeans[:,ind_nonan]
#optionally standardize
if stand_for_kmeans == 'space':
    print('INFO: error patterns are standarized in space prior to applying kmeans clustering...')
    ##standardization with numpy
    mn_step = np.tile(np.mean(arr_error_2_kmeans_nonans,axis=1),(arr_error_2_kmeans_nonans.shape[1],1)).transpose()
    std_step = np.tile(np.std(arr_error_2_kmeans_nonans,axis=1),(arr_error_2_kmeans_nonans.shape[1],1)).transpose()
    arr_error_2_kmeans_nonans = (arr_error_2_kmeans_nonans - mn_step)/std_step
    ##standarization with StandardScaler, see https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters
    #scaler = StandardScaler()
    #arr_error_2_kmeans_nonans = np.transpose(scaler.fit_transform(np.transpose(arr_error_2_kmeans_nonans)))
elif stand_for_kmeans == 'models':
    print('INFO: error patterns are standarized along the model axis prior to applying kmeans clustering...')
    ##standardization with numpy
    mn_step = np.tile(np.mean(arr_error_2_kmeans_nonans,axis=0),(arr_error_2_kmeans_nonans.shape[0],1))
    std_step = np.tile(np.std(arr_error_2_kmeans_nonans,axis=0),(arr_error_2_kmeans_nonans.shape[0],1))
    arr_error_2_kmeans_nonans = (arr_error_2_kmeans_nonans - mn_step)/std_step
elif stand_for_kmeans == 'no':
    print('INFO: GCM errors are not standardized prior to applying k-means clustering...')
else:
    raise Exception('Error: unknown entry for <stand_for_kmeans>!')

#perform a PCA to reduce the spatial dimensions of the error patterns, see https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
print('INFO: PCA is applied prior to kmeans clustering...') 
pca_setup = PCA(n_components=np.min(arr_error_2_kmeans_nonans.shape))
principalComponents = pca_setup.fit(arr_error_2_kmeans_nonans)
exp_var = principalComponents.explained_variance_ratio_.cumsum()
retain_ind = np.min(np.where(exp_var > exp_var_thresh))
print('INFO: '+str(retain_ind+1)+' PCs explain '+str(exp_var_thresh*100)+' percent of the total spatial variance. Error data is now replaced by PCs and dimensions are reduced accordingly...')
arr_error_2_kmeans_nonans = principalComponents.components_[:,0:retain_ind]

#set architecture of the kmeans algorithm and fit
print('INFO: Kmeans clustering will be applied for '+str(km_nr)+' clusters, '+str(km_n_init)+'_inital centroids, '+str(km_max_iter)+' maximum iterations, '+str(km_tol)+' tolerance and random state set to '+str(km_random_state))
km = KMeans(n_clusters=km_nr, init=km_init, n_init=km_n_init, max_iter=km_max_iter,tol=km_tol, random_state=km_random_state, verbose=0)
clusters = km.fit(arr_error_2_kmeans_nonans)
pred_clusters = km.predict(arr_error_2_kmeans_nonans)

#get elbow diagram and Silhuette scores 
distortions = []
silhouette_coefficients = []
for i in range(km_nr):
    print('INFO: fitting kmeans clustering for '+str(i+1)+' centroids....')
    km_step = KMeans(n_clusters=i+1, init=km_init, n_init=km_n_init, max_iter=km_max_iter, tol=km_tol, random_state=km_random_state)
    km_step.fit(arr_error_2_kmeans_nonans)
    distortions.append(km_step.inertia_)
    print('INFO: '+str(km_step.n_iter_)+' iterations were required for fitting the centroids.')
    if i > 0:
        sil_score = metrics.silhouette_score(arr_error_2_kmeans_nonans, km_step.labels_)
        silhouette_coefficients.append(sil_score)
        print('Silhouette score is: '+str(sil_score))
    #preds = km_step.fit_predict(arr_error_2_kmeans_nonans) #https://stackoverflow.com/questions/51138686/how-to-use-silhouette-score-in-k-means-clustering-from-sklearn-library
    #silhuette = metrics.silhouette_score(arr_error_2_kmeans_nonans, preds)
    #print('INFO: The Silhuette score for '+str(i)+' clusters is: '+str(silhuette))

#plot the elbow diagram
fig = plt.figure()
plt.plot(range(1, km_nr+1), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of the squared errors')
plt.xticks(range(1, km_nr+1))
savepath_elbow = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/elbow_nrclust_'+str(km_nr)+'_stand_'+stand_for_kmeans+'_'+corrmethod+'_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savepath_elbow, dpi=dpival)
plt.close('all')

#plot the Silhouette Coefficients
fig = plt.figure()
plt.plot(range(2, km_nr+1), silhouette_coefficients, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette coefficient')
plt.xticks(range(1, km_nr+1))
savepath_elbow = figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/silhouette_nrclust_'+str(km_nr)+'_stand_'+stand_for_kmeans+'_'+corrmethod+'_'+errortype+'_wrt_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
plt.savefig(savepath_elbow, dpi=dpival)
plt.close('all')

#generate a scatterplot for the relationship between average model performance and spatial error correlation
fig = plt.figure()
ax = fig.gca()
rho_meanperf_meanpattcorr = stats.pearsonr(mean_error,mean_corr)
rho_medperf_medpattcorr = stats.pearsonr(median_error,median_corr)
rk_meanperf_meanpatt = stats.spearmanr(mean_error,mean_corr,nan_policy='omit')
rk_medperf_medpatt = stats.spearmanr(median_error,median_corr,nan_policy='omit')
for ii in list(range(len(median_error))):
    ax.plot(mean_error[ii],mean_corr[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0)
ax.patch.set_edgecolor('black') 
ax.patch.set_linewidth(1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Mean '+errortype,size=12)
plt.ylabel('Mean spatial correlation of the grid-box-scale '+errortype,size=12)
plt.title('r = '+str(round(rho_meanperf_meanpattcorr[0],2)))
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig(figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/mean_'+errortype+'_vs_mean_corr_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat)
plt.close('all')

#organize output of the kmeans clustering
model2clusters = pd.DataFrame(data=model,index=clusters.labels_)

centroid_models = []
#get model nearest to each centroid
for cc in list(range(km_nr)):
    cent_mat = np.tile(clusters.cluster_centers_[cc,:],(len(model),1))
    diff = np.sum(arr_error_2_kmeans_nonans - cent_mat,axis=1)
    cent_ind = np.argmin(np.abs(diff))
    centroid_models.append(model[cent_ind])

#plot example barplot
if plot_freq == 'yes':
    mod_sample = arr_mod_freq[:,5,5,-1]
    obs_sample = obs_freq[:,5,5]
    joint_sample = np.stack((obs_sample,mod_sample),axis=0)
    fig = plt.figure()
    width = 0.3
    plt.bar(np.arange(1,len(mod_sample)+1)+width,mod_sample,width=width,label='GCM')
    plt.bar(np.arange(1,len(obs_sample)+1),obs_sample,width=width,label='Reanalysis')
    plt.xticks(np.arange(1,len(obs_sample)+1))
    plt.xlabel('Lamb Weather Type')
    plt.ylabel('Relative Frequency')
    plt.legend()
    plt.savefig(figpath+'/'+figfolder+'/'+region+'/'+season_label.lower()+'/example_barplot_'+refdata+'_'+region+'_'+groupby+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat)
    plt.close('all')

#map multi-mode median performance
cbar_median_error = 'Multi-model median '+errortype
title_median_error = 'Median '+errortype+' for '+str(len(model))+' models w.r.t '+refdata+'_'+region+' ruout '+correct_ru+' altruns '+alt_runs+' '+str(taryears[0])+'-'+str(taryears[1])+' '+season_label
savename_median_error = savedir_error_map+'/median_'+errortype+'_'+str(len(model))+'_models_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
draw_error_map('error',region,lats_values,lons_values,np.transpose(np.nanmedian(arr_error,axis=2)),colormap_error,halfres,cbounds_map,snapval,savename_median_error,title_median_error,cbar_median_error,figformat,textsize,dpival,norm=None,ticks_cbar=None)
#map multi-model iqr of the performance
cbar_iqr_error = 'Multi-model IQR of '+errortype
title_iqr_error = 'IQR of '+errortype+' for '+str(len(model))+' models w.r.t '+refdata+'_'+region+' ruout '+correct_ru+' altruns '+alt_runs+' '+str(taryears[0])+'-'+str(taryears[1])+' '+season_label
savename_iqr_error = savedir_error_map+'/iqr_'+errortype+'_'+str(len(model))+'_models_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+season_label.lower()+'.'+figformat
q3, q1 = np.nanpercentile(arr_error, [75 ,25],axis=2)
draw_error_map('error',region,lats_values,lons_values,np.transpose(q3-q1),colormap_error,halfres,cbounds_map/2,snapval,savename_iqr_error,title_iqr_error,cbar_iqr_error,figformat,textsize,dpival,norm=None,ticks_cbar=None)

print('INFO: analysis_hist.py has been run successfully!')

