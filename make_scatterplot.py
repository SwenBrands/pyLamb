# -*- coding: utf-8 -*-
"""
This script is used to make scatterplots on the basis of results previously generated with analysis_hist.py
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
from sklearn import preprocessing
import os
import seaborn as sns
import pdb #then type <pdb.set_trace()> at a given line in the code below
plt.style.use('seaborn')

exec(open('get_historical_metadata.py').read())

#set general input parameters
home = os.getenv('HOME')
input_path = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs'
output_path = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs/summary_scatterplots'
figformat = 'pdf' #format of output figures, pdf or png
mode = 'median' #statistic to be plotted, median, permodel or all
standardize = 'logstand' #no, mean, stand, logstand #optionally remove the 1) mean, 2) standardize or 3) log-transform and standardize the median erros; these options are only applicable if mode = 'median'

#x-axis configuration
refrean_x = 'era5' #interim, jra55 or era5 reference reanlaysis dataset
region_x = 'escena' #nh, sh, eurocordex, cordexna, escena
errortype_x = 'MAE' #error type to be computed, MAE or KL, Mean Absolute Error or Kullback-Leibler divergence
season_x = 'JJAS' #season to be plotted
classes_needed_x = 27 #minimum nuber of classes required to plot the result on a map, 27 for NH and 20 for SH
minfreq_x = 0.001 #minimum frequency required to plot the results on a map (in decimals), 0.001 in gmd-2020-418
groupby_x = 'agcm'
correct_ru_x = 'no'
alt_runs_x = 'no'

#y-axis configuration
refrean_y = 'era5' #interim, jra55 or era5 reference reanlaysis dataset
region_y = 'escena' #nh, sh, eurocordex, cordexna, escena
errortype_y = 'PERSall' #error type to be computed, MAE or KL, Mean Absolute Error or Kullback-Leibler divergence
season_y = 'JJAS' #season to be plotted
classes_needed_y = 27 #minimum nuber of classes required to plot the result on a map, 27 for NH and 20 for SH
minfreq_y = 0.001 #minimum frequency required to plot the results on a map (in decimals), 0.001 in gmd-2020-418
groupby_y = 'agcm'
correct_ru_y = 'no'
alt_runs_y = 'no'

#EXECUTION####################
print('X axis configuration is '+refrean_x+', '+region_x+', '+errortype_x+', '+str(classes_needed_x)+', '+str(minfreq_x)+', '+groupby_y+', '+correct_ru_x)
print('Y axis configuration is '+refrean_y+', '+region_y+', '+errortype_y+', '+str(classes_needed_y)+', '+str(minfreq_y)+', '+groupby_y+', '+correct_ru_y)
print('The figure output format is '+figformat)

#check consitstency of input parameters
if region_x != region_y:
    if mode != 'median':
        raise Exception('ERROR: If <region_x> and <region_y> are different, <mode> must be set to <median> !')

#if standardize == 'yes' and mode != 'median':
 #   raise Exception('ERROR: standardization is not applicable for mode = '+mode+' in this version of the script!')

#define input paths and load the data
file_x = input_path+'/figs_ref'+refrean_x+'/'+region_x+'/'+season_x+'/gridbox_errors_'+errortype_x+'_wrt_'+refrean_x+'_'+region_x+'_'+groupby_x+'_ruout_'+correct_ru_x+'_altruns_'+alt_runs_x+'_1979-2005.nc'
file_y = input_path+'/figs_ref'+refrean_y+'/'+region_y+'/'+season_y+'/gridbox_errors_'+errortype_y+'_wrt_'+refrean_y+'_'+region_y+'_'+groupby_y+'_ruout_'+correct_ru_y+'_altruns_'+alt_runs_y+'_1979-2005.nc'
nc_x = xr.open_dataset(file_x)
nc_y = xr.open_dataset(file_y)
model_x = nc_x.coords['gcm'].values.tolist()
model_y = nc_y.coords['gcm'].values.tolist()

#check if the two model lists are identical
if model_x == model_y:
    print('INFO: the two gcm lists model_x and model_y are identical.')
    model = model_x
else:
    raise Exception('ERROR: model_x and model_y do not agree!')

#get raw errros and calculate median error values
error_x = nc_x[errortype_x].values
error_y = nc_y[errortype_y].values
median_x = np.nanmedian(error_x,axis=0)
median_y = np.nanmedian(error_y,axis=0)
mean_x = np.nanmean(error_x,axis=0)
mean_y = np.nanmean(error_y,axis=0)
if standardize == 'logstand':
    print('Info: median errors are log transformed (since the were found to be positively skewed) and then standardized...')
    median_x = preprocessing.scale(np.log(median_x))
    median_y = preprocessing.scale(np.log(median_y))
elif standardize == 'mean':
    print('Info: median errors are transformed to anomalies...')
    median_x = preprocessing.scale(median_x - mean_x)
    median_y = preprocessing.scale(median_y - mean_y)
elif standardize == 'stand':
    print('Info: median errors are standardized...')
    median_x = preprocessing.scale(median_x)
    median_y = preprocessing.scale(median_y)
elif standardize == 'no':
    print('As requested by the user, the domain-wide median error is not standardized prior to visualization !')
else:
    raise Exception('ERROR: unknown entry for <standardize>!')

#initialize lists containing the metdata for each model, which is assigned below in <get_historical_metadata,py>
mrun = [' '] * len(model) #run specifications
doi = [' ']  * len(model) #dois
atmos = [' ']  * len(model) #GCM
surface = [' ']  * len(model) #land surface model
ocean = [' ']  * len(model) #OGCM
seaice = [' ']  * len(model) #sea-ice model
aerosols = [' ']  * len(model) #aerosols
chemistry = [' ']  * len(model) #atmospheric chemistry
obgc = [' ']  * len(model) #ocean biogeochemistry
landice = [' ']  * len(model) #ice sheets
complexity = [' '] * len(model) # model complexity code complexity, string
addinfo = [' ']  * len(model) #additional info
family = [' '] * len(model) #complexity family, currently "esm" or "gcm"
cmip = np.zeros(len(model)) #cmip generation, 5 or 6
rgb = [' '] * len(model) #rgb code for each model family
marker = [' '] * len(model) #marker for each model of a given family
latres_atm = [' '] * len(model) #number of meridional grid-boxes
lonres_atm = [' '] * len(model) #number of zonal grid-boxes at the equator (due to the presence of Gaussian reduced grids)
lev_atm = [' '] * len(model) #number of vertical levels in the atmosphere
latres_oc = [' '] * len(model) #as above, but for the ocean
lonres_oc = [' '] * len(model) # "
lev_oc = [' '] * len(model) # "
ecs = [' '] * len(model) #equilibrium climate sensitivity for each model
tcr = [' '] * len(model) #transient climate response for each model

#assign metadata to each model
for mm in range(len(model)):
    mrun[mm],complexity[mm],family[mm],cmip[mm],rgb[mm],marker[mm],latres_atm[mm],lonres_atm[mm],lev_atm[mm],latres_oc[mm],lonres_oc[mm],lev_oc[mm],ecs[mm],tcr[mm] = get_historical_metadata(model[mm])

rgb_unique = np.unique(rgb)

output_dir = output_path+'/'+errortype_x+'_vs_'+errortype_y+'/'+refrean_x+'_vs_'+refrean_y+'/'+region_x+'_vs_'+region_y
if os.path.isdir(output_dir) != True:
    os.makedirs(output_dir)
#make the scatterplot
if mode == 'median':
    #calculate correlation coefficient allowing nans
    x_pd = pd.DataFrame(median_x)
    y_pd = pd.DataFrame(median_y)
    rho = round(x_pd.corrwith(y_pd,drop=True)[0],3)
    fig = plt.figure()
    ax = fig.gca()
    for ii in list(range(len(median_x))):
        ax.plot(median_x[ii],median_y[ii],color=rgb[ii],marker=marker[ii],markersize=10,markeredgecolor='k',markeredgewidth=1.0,linestyle='None')
    #ax.set_aspect(1./ax.get_data_ratio())
    #ax.set_aspect('equal', 'box')
    ax.patch.set_edgecolor('black') 
    ax.patch.set_linewidth(1)
    ax.set_aspect(1./ax.get_data_ratio())
    plt.title('Domain-wide median error for all GCMs, rho:'+str(rho))
    x_plus_y_values = np.concatenate((median_x,median_y))
    addval = np.max(x_plus_y_values)/30
    #define x and y limits
    if standardize in ('logstand','mean','stand'):
        plt.xlim(np.min(x_plus_y_values) - addval, np.max(x_plus_y_values) + addval)
        plt.ylim(np.min(x_plus_y_values) - addval, np.max(x_plus_y_values) + addval)
        ax.plot(median_x,median_x,'k-')
    elif standardize == 'no':
        print('X and Y limits are no specifically set for <standardize = no>.')
    else:
        raise Exception('ERROR: check entry for <standardize> parameter !')
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    plt.xlabel(refrean_x.upper()+', '+region_x.upper()+', '+errortype_x.upper(),size=12)
    plt.ylabel(refrean_y.upper()+', '+region_y.upper()+', '+errortype_y.upper(),size=12)
    #plt.axis('tight')
    savename = output_dir+'/scatterplot_allgcms_'+mode+'_'+season_x+'_'+season_y+'_'+refrean_x+'_vs_'+refrean_y+'_'+region_x+'_vs_'+region_y+'_'+errortype_x+'_vs_'+errortype_y+'_stand_'+standardize+'.'+figformat
    plt.savefig(savename)
    plt.close('all')
elif mode == 'all':
    fig = plt.figure()
    ax = fig.gca()
    for ii in list(range(len(median_x))):
        ax.plot(error_x[:,ii],error_y[:,ii],color=rgb[ii],marker='o',markersize=3,markeredgecolor='None',linestyle='None')
    ax.plot(median_x,median_x,'k-')
    ax.set_aspect(1./ax.get_data_ratio())
    ax.patch.set_edgecolor('black') 
    ax.patch.set_linewidth(1)
    #plt.title(refrean_x+', '+region_x+', '+errortype_x+', vs '+refrean_y+', '+region_y+', '+errortype_y)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(refrean_x.upper()+', '+region_x.upper()+', '+errortype_x.upper(),size=12)
    plt.ylabel(refrean_y.upper()+', '+region_y.upper()+', '+errortype_y.upper(),size=12)
    plt.savefig(output_dir+'/scatterplot_allgcms_'+mode+'_'+season_x+'_'+season_y+'_'+refrean_x+'_vs_'+refrean_y+'_'+region_x+'_vs_'+region_y+'_'+errortype_x+'_vs_'+errortype_y+'.'+figformat)
    plt.close('all')
elif mode == 'permodel':
    rho_all_models = np.zeros(len(model))
    for ii in list(range(len(model))):
        #calculate correlation coefficient allowing nans
        x_pd = pd.DataFrame(error_x[:,ii])
        y_pd = pd.DataFrame(error_y[:,ii])
        if standardize == 'yes':
            #x_pd = np.log(x_pd)
            #y_pd = np.log(y_pd)
            x_pd = pd.DataFrame((x_pd.values - np.nanmean(x_pd.values)) / np.nanstd(x_pd.values))
            y_pd = pd.DataFrame((y_pd.values - np.nanmean(y_pd.values)) / np.nanstd(y_pd.values))
        rho = round(x_pd.corrwith(y_pd,drop=True)[0],3)
        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.plot(error_x[:,ii],error_y[:,ii],color=rgb[ii],marker='o',markersize=4,markeredgecolor='k',markeredgewidth=0.5,linestyle='None')
        if standardize == 'yes':
            plt.plot([0,2],[0,2],'k-')
            maxval = np.nanmax(np.concatenate((x_pd,y_pd)))
            minval =  np.nanmin(np.concatenate((x_pd,y_pd)))
            #plt.xlim(minval,maxval)
            #plt.ylim(minval,maxval)
            #ax.axes('equal')
        #ax.set_aspect(1./ax.get_data_ratio())
        #ax.patch.set_edgecolor('black') 
        #ax.patch.set_linewidth(1)
        plt.title('Grid-box-scale errors for '+model[ii]+', rho:'+str(rho))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(refrean_x.upper()+', '+region_x.upper()+', '+errortype_x.upper(),size=12)
        plt.ylabel(refrean_y.upper()+', '+region_y.upper()+', '+errortype_y.upper(),size=12)
        fig.tight_layout()
        plt.savefig(output_dir+'/scatterplot_'+model[ii]+'_'+mode+'_'+season_x+'_'+season_y+'_'+refrean_x+'_vs_'+refrean_y+'_'+region_x+'_vs_'+region_y+'_'+errortype_x+'_vs_'+errortype_y+'.'+figformat)
        plt.close('all')
        rho_all_models[ii] = rho
        rho_all_models = pd.DataFrame(data=rho_all_models,index=model)
else:
    raise Exception('ERROR: check entry for <mode>')
