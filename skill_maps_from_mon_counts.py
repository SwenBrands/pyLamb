# -*- coding: utf-8 -*-

''''This script calculates and plots Lamb Weather Type time-series and signal-to-noise ratios at specified locations'''

#load packages
import numpy as np
import pandas as pd
import xarray as xr
from scipy import fft, arange, signal
import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import os
from scipy.stats import norm
from scipy.stats import t
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm
import xskillscore as xs
import seaborn as sns
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('analysis_functions.py').read())
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)

#set input parameter
obs = 'era5' #cera20c or mpi_esm_1_2_hr or ec_earth3
ensemble = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3'] #cera20c or mpi_esm_1_2_hr or ec_earth3
experiment = ['dcppA','dcppA','dcppA','historical'] #historical, amip, piControl, 20c or dcppA, used to load the data
experiment_out = ['dcppA_1','dcppA_5','dcppA_10','historical'] #used as label in the xarray data array produced here; combines experiment with lead time if indicated
lead_time = [1,5,10,None] #lead time or forecast year or the dcppA LWT data
start_years = [1961,1965,1970,1961] #list containing the start years of the experiment defined in <experiment>
end_years = [2019,2023,2028,2028] #list containing the end years of the experiment defined in <experiment>

ensemble_color = ['orange','black','grey','blue']
ensemble_linestyle = ['dashed','dashed','solid','dotted']

reference_period = [1970,2014] # "from_data" or list containing the start and end years
# seasons = ['ONDJFM','AMJJAS','DJF','JJA'] #list of seasons to be considered: year, DJF, MAM, JJA or SON
# months = [[10,11,12,1,2,3],[4,5,6,7,8,9],[12,1,2],[1,2,3]] #list of months corresponding to each season

seasons = ['ONDJFM','AMJJAS'] #list of seasons to be considered: year, DJF, MAM, JJA or SON
months = [[10,11,12,1,2,3],[4,5,6,7,8,9]] #list of months corresponding to each season

# seasons = ['ONDJFM'] #list of seasons to be considered: year, DJF, MAM, JJA or SON
# months = [[10,11,12,1,2,3]] #list of months corresponding to each season

tarwts = [['PA'],['DANE','PDNE','DCNE'],['DAE','PDE','DCE'], ['DASE','PDSE','DCSE'], ['DAS','PDS','DCS'], ['DASW','PDSW','DCSW'], ['DAW','PDW','DCW'], ['DANW','PDNW','DCNW'], ['DAN','PDN','DCN'], ['PC'], ['U']] #original names for 11 types
tarwts_name = ['PA','NE','E','SE','S','SW','W','NW','N','PC','U'] #summarized names for 11 types

# tarwts = [['PA'],['DANE','PDNE','DCNE']] #original names for 11 types
# tarwts_name = ['PA','NE'] #summarized names for 11 types

# tarwts = [['PA']] #original names for 11 types
# tarwts_name = ['PA'] #summarized names for 11 types

#set directory paths
figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs/ec_earth3' #base path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2'
#add_functions_root = '/lustre/gmeteo/PTICLIMA/Scripts/SBrands/pyPTIclima/pySeasonal' #path to additional functions used by pySeasonal
rundir = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb'

#set options for statistical analysis
center_wrt = 'memberwise_mean' # ensemble_mean or memberwise_mean; centering w.r.t. to ensemble (or overall) mean value or member-wise temporal mean value prior to calculating signal-to-noise
meanperiod = 10 #running-mean period in years
std_critval = 1.28 #1 = 68%, 1.28 = 80 %, 2 = 95%; standard deviation used to define the critical value above or below which the signal-to-noise ratio is assumed to be significant.
rho_ref = '20c_era5' #20c_era5 or 20c_cera20c; reference observational dataset used to calculate correlations. Must be included in the <experiment_out> input parameter defined above
detrending = 'yes' #remove linear trend from LWT count time series, yes or no
test_level = 95 #test level in %, following the nomenclature used in Wilks 2006; used to obtaine the critical values for the RPC
repetitions = 100 #number of repetitions used for Monte-Carlo significance testing
anom = 'yes' #yes or no; calculate correlation measures and RPC on anomalies or raw values, respectively
exclude_members = 3 #number of members to be excluded for Monte-Carlo significance testing
rho_critval = 0.279 #critical values for a significant Pearson correlation coefficient for n = 50, alpha=0.05, two-tailed test

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
yearly_units = '%' # count, % (relative frequency) or z-score; unit of the yearly LWT counts

##visualization options
map_proj_nh = ccrs.NorthPolarStereo() #ccrs.PlateCarree()
map_proj_sh = ccrs.SouthPolarStereo() #ccrs.PlateCarree()
axes_config = 'equal' # equal or individual; if set to equal, the axes limits are equal for all considered lead-times in a given city; if set to individual, the axes are optimized for each specific lead-time in that city
plot_sig_stn_only = 'no' #plot only significant signal-to-noise ratios in pcolor format, yes or no
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures
colormap = 'bwr'
edgecolor = 'black' #if set to any colour, e.g. black, cell boundaries are depicted in the pcolor plots generated by this script.

##execute ###############################################################################################
os.chdir(rundir)

if len(tarwts) != len(tarwts_name):
    raise Exception('ERROR: the lenght of <tarwts> must equal the lenght of <tarwts_name> !')

timestep = 'mon'
taryears = [np.max(start_years),np.min(end_years)] #years common to all experiments defined in <experiment>

ref_period = reference_period
print('The reference period used for anomaly calculation is '+str(ref_period))

# wtnames = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
# wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

study_years = np.arange(np.array(taryears).min(),np.array(taryears).max()+1,1) #numpy array of years limited by the lower and upper end of considered years
t_startindex = int(meanperiod/2) #start index of the period to be plotted along the x axis
t_endindex = int(len(study_years)-(meanperiod/2-1)) #end index of the period to be plotted along the x axis

##loop through all LWTs, seasons experiments and members of a given model experiment (historical and dcppA with different forecast years)
for lwt in np.arange(len(tarwts)):
    for sea in np.arange(len(seasons)):
        seaslabel = seasons[sea]
        ##loop throught the lead-times, lt
        for en in np.arange(len(experiment)):
            print('Aggregating '+str(tarwts[lwt])+' type(s) labelled as '+tarwts_name[lwt]+' for season '+seasons[sea]+' and experiment '+experiment[en]+' with lead-time '+str(lead_time[en])+' over the NH and SH...')
            #get ensemble configuration as defined in analysis_functions,py, see get_ensemble_config() function therein
            model,mrun,model_label,tarhours = get_ensemble_config(ensemble[en],experiment[en])
            
            print('INFO: get monthly LWT counts for '+ensemble[en]+' with '+str(len(mrun))+' runs, lead time '+str(lead_time[en])+', season '+str(seasons[sea])+', start year '+str(start_years[en])+' and end year '+str(end_years[en]))
           
            for mm in list(range(len(model))):
                #get metadata for this GCM, incluing start and end year label of the LWT input files to be loaded below
                if experiment[en] in ('amip','dcppA','historical','piControl'):
                    runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm]) #check whether historical GCM configurations agree with those used in DCPPA ! 
                    file_taryears, timestep_h = get_target_period(model[mm],experiment[en],cmip_f=cmip,lead_time_f=lead_time[en]) #timestep_h is the temporal resolution of the instantaneous LWT file underlying the monthly counts obtained with makecalcs_parallel_plus_counts.py
                elif experiment[en] == '20c':
                    print('get_historical_metadata.py is not called for experiment = '+experiment[en]+'...')
                    file_taryears, timestep_h = get_target_period(model[mm],experiment[en])
                else:
                    raise Exception('ERROR: unknown entry in <experiment> input parameter')
                
                file_startyear = file_taryears[0]
                file_endyear = file_taryears[1]
                
                store_wt_nh = store_wt_orig+'/'+timestep+'/'+experiment[en]+'/nh'
                store_wt_sh = store_wt_orig+'/'+timestep+'/'+experiment[en]+'/sh'
                if experiment[en] == 'dcppA':
                    wt_file_nh = store_wt_nh+'/wtcount_mon_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_nh_'+str(lead_time[en])+'y_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the NH LWT catalogues
                    wt_file_sh = store_wt_sh+'/wtcount_mon_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_sh_'+str(lead_time[en])+'y_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the SH LWT catalogues
                elif experiment[en] == 'historical':
                    wt_file_nh = store_wt_nh+'/wtcount_mon_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_nh_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the NH LWT catalogues
                    wt_file_sh = store_wt_sh+'/wtcount_mon_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_sh_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the SH LWT catalogues
                else:
                    raise Exception('ERROR: check entry for experiment[en] !')
                #load the LWT time series for the centerlat and target years obtained above, merge them and close hemispheric files
                wt_nh = xr.open_dataset(wt_file_nh)
                wt_sh = xr.open_dataset(wt_file_sh)
                wt = xr.merge([wt_nh,wt_sh])
                wt_nh.close()
                wt_sh.close()
                del(wt_nh,wt_sh)
                
                #select requested season
                if seasons[sea] in ('MAM','JJA','SON'):
                    wt = wt.sel(time=(wt['time'].dt.season == seasons[sea]))
                    print('Processing '+seasons[sea]+' season...')
                elif seasons[sea] in ('DJF','ONDJFM','AMJJAS'): #see https://stackoverflow.com/questions/70658388/how-to-select-djfm-season-instead-of-xarray-groupby
                    print('Processing '+seasons[sea]+' season...')
                    # wt = wt.isel(time=wt.time.dt.month.isin(months[sea])) #get target months
                    # wt = wt.rolling(time=len(months[sea])).sum() #rolling sum
                    # wt = wt.isel(time=wt.time.dt.month == months[sea][-1]) #get the accumulated values ending in February
                    wt = get_seasonal_mean(wt,months[sea])
                elif seasons[sea] == 'year':
                    print('For season[sea] = '+seasons[sea]+', the entire calendar year will be considered.')
                else:
                    raise Exception('ERROR: check entry for <seasons[sea]> !')
                
                #select requested years
                dates_wt = pd.DatetimeIndex(wt.time.values)
                #year_ind_wt = np.where((dates_wt.year >= taryears[0]) & (dates_wt.year <= taryears[1]))[0]
                year_bool = (dates_wt.year >= taryears[0]) & (dates_wt.year <= taryears[1])
                wt = wt.isel(time=year_bool)
                
                #retain requested LWTs, sum along lwt dimension to get the joint counts per month
                wt = wt.sel(lwt=tarwts[lwt]).sum(dim='lwt')
                
                ##calculate year-to-year relative frequencies (%)
                wt = wt.groupby('time.year').sum('time').rename({'year':'time'})
                wt.counts[:] = wt.counts / (wt.days_per_month*int(timestep_h[0]))*100
                
                #optional linear detrending
                if detrending == 'yes':
                    print('INFO: As requested by the user, the year-to-year model LWT count time series are linearly detrended.')
                    wt.counts[:] = lin_detrend(wt.counts,'no')
                elif detrending == 'no':
                    print('INFO: As requested by the user, the year-to-year LWT count time series are not detrended.')
                else:
                    raise Exceptions('ERROR: check entry for <detrending> input parameter !')
                
                #init numpy array to be filled with year-to-year relative LWT frequencies, loaded experiments and lead times
                if lwt == 0 and sea == 0 and en == 0 and mm == 0:
                    taryears_index = np.arange(taryears[0],taryears[1]+1) 
                    wt_rf_np = np.zeros((len(tarwts),len(seasons),len(experiment),len(model),len(taryears_index),len(wt.lon),len(wt.lat)))
                    experiment_loaded = []
                    lead_time_loaded = []
                    #dimensions needed to convert wt_rf_np into xarray data array format
                    years = wt.time
                    lon = wt.lon
                    lat = wt.lat
                
                #assign
                wt_rf_np[lwt,sea,en,mm,:,:,:] = wt.counts.values
                
                #clean up
                wt.close()
                del(wt)
            
#convert to xarray data array
run_index = np.arange(len(mrun))
wt_mod = xr.DataArray(wt_rf_np, coords=[tarwts_name,seasons, experiment_out, run_index, years, lon, lat ], dims=['lwt','season','experiment','run_index','time','lon','lat'], name='counts')
wt_mod.attrs['units'] = 'relative frequency per year (%)'

#load observations
for lwt in np.arange(len(tarwts)):
    for sea in np.arange(len(seasons)):
        store_wt_nh_obs = store_wt_orig+'/'+timestep+'/historical/nh'
        store_wt_sh_obs = store_wt_orig+'/'+timestep+'/historical/sh'
        wt_file_nh_obs = store_wt_nh_obs+'/wtcount_mon_era5_historical_r1i1p1_nh_1940_2022.nc' #path to the NH LWT catalogues
        wt_file_sh_obs = store_wt_sh_obs+'/wtcount_mon_era5_historical_r1i1p1_sh_1940_2022.nc' #path to the SH LWT catalogues

        wt_nh_obs = xr.open_dataset(wt_file_nh_obs)
        wt_sh_obs = xr.open_dataset(wt_file_sh_obs)
        wt_obs = xr.merge([wt_nh_obs,wt_sh_obs])
        wt_nh_obs.close()
        wt_sh_obs.close()
        del(wt_nh_obs,wt_sh_obs)

        #select requested season as above for model data
        if seasons[sea] in ('MAM','JJA','SON'):
            wt_obs = wt_obs.sel(time=(wt_obs['time'].dt.season == seasons[sea]))
            print('Processing '+seasons[sea]+' season...')
        elif seasons[sea] in ('DJF','ONDJFM','AMJJAS'): #see https://stackoverflow.com/questions/70658388/how-to-select-djfm-season-instead-of-xarray-groupby
            print('Processing '+seasons[sea]+' season...')
            # wt_obs = wt_obs.isel(time=wt_obs.time.dt.month.isin([12,1,2])) #get DJF months
            # wt_obs = wt_obs.rolling(time=3).sum() #rolling sum
            # wt_obs = wt_obs.isel(time=wt_obs.time.dt.month == 2) #get the accumulated values ending in February
            # wt_obs = wt_obs.isel(time=wt_obs.time.dt.month.isin(months[sea])) #get target months
            # wt_obs = wt_obs.rolling(time=len(months[sea])).sum() #rolling sum
            # wt_obs = wt_obs.isel(time=wt_obs.time.dt.month == months[sea][-1]) #get the accumulated values ending in February
            wt_obs = get_seasonal_mean(wt_obs,months[sea])
        else:
            raise Exception('ERROR: check entry for <seasons[sea]> !')

        #select requested years
        dates_wt_obs = pd.DatetimeIndex(wt_obs.time.values)
        year_bool_obs = (dates_wt_obs.year >= taryears[0]) & (dates_wt_obs.year <= taryears[1])
        wt_obs = wt_obs.isel(time=year_bool_obs)

        #retain requested LWTs, sum along lwt dimension to get the joint counts per month
        wt_obs = wt_obs.sel(lwt=tarwts[lwt]).sum(dim='lwt')

        #calculate year-to-year relative frequencies (%)
        wt_obs = wt_obs.groupby('time.year').sum('time').rename({'year':'time'})
        wt_obs.counts[:] = wt_obs.counts / (wt_obs.days_per_month*int(timestep_h[0]))*100
        
        #optional linear detrending
        if detrending == 'yes':
            print('INFO: As requested by the user, the year-to-year observational LWT count time series are linearly detrended.')
            wt_obs.counts[:] = lin_detrend(wt_obs.counts,'no')
        elif detrending == 'no':
            print('INFO: As requested by the user, the year-to-year LWT count time series are not detrended.')
        else:
            raise Exceptions('ERROR: check entry for <detrending> input parameter !')
        
        #init numpy array to be filled with year-to-year relative LWT frequencies, loaded experiments and lead times
        if lwt == 0 and sea == 0:
            wt_rf_np_obs = np.zeros((len(tarwts),len(seasons),len(taryears_index),len(wt_obs.lon),len(wt_obs.lat)))
            #dimensions needed to convert wt_rf_np into xarray data array format
            years = wt_obs.time
            lon = wt_obs.lon
            lat = wt_obs.lat
        
        #assign
        wt_rf_np_obs[lwt,sea,:,:,:] = wt_obs.counts.values
        
        #clean up
        wt_obs.close()
        del(wt_obs)

#convert to xarray data array
wt_obs = xr.DataArray(wt_rf_np_obs, coords=[tarwts_name,seasons, years, lon, lat ], dims=['lwt','season','time','lon','lat'], name='counts')
wt_obs.attrs['units'] = 'relative frequency per year (%)'
wt_obs_tmean = wt_obs.mean(dim='time').rename('temporal_mean_count') 
wt_obs_tmean = xr.concat([wt_obs_tmean] * len(wt_obs.time), dim='time') #replicate temporal mean values along the new <time> dimension
wt_obs_tmean = xr.DataArray(wt_obs_tmean.transpose('lwt','season','time','lon','lat').values,coords=wt_obs.coords,dims=wt_obs.dims,name='observed_temporal_mean_count')
wt_obs_anom = (wt_obs - wt_obs_tmean).rename('observed_count_anomaly') 

#calculate signal-to-noise ratios based on anomalies, as described in https://doi.org/10.1007/s00382-010-0977-x
wt_mod_tmean = wt_mod.mean(dim='time').rename('memberwise_temporal_mean_count') #calcualte memberwise temporal-mean values
wt_mod_tmean = xr.concat([wt_mod_tmean] * len(wt_mod.time), dim='time') #replicate temporal mean values along the new <time> dimension
#redefine the xr data array with the right order of coordinates
wt_mod_tmean = xr.DataArray(wt_mod_tmean.transpose('lwt','season','experiment','run_index','time','lon','lat').values,coords=wt_mod.coords,dims=wt_mod.dims,name='memberwise_temporal_mean_count')

##alternative method to replicate the temporal mean value along the time dimension / coordinate which, however, leads to a wrong order of the coordinates listed in the xr data array and is thus discarded
#wt_mod_tmean = wt_mod_tmean.assign_coords(time=wt_mod.time) #add coordinate <time>
#wt_mod_tmean = wt_mod_tmean.assign_coords({'time' : wt_mod.time}) #alternative option to add coordinate <time>
#wt_mod_tmean = wt_mod_tmean.transpose('lwt','season','experiment','run_index','time','lon','lat') #reorder dimensions to agree with <wt_mod>

wt_mod_anom = (wt_mod - wt_mod_tmean).rename('memberwise_count_anomaly') #calcualte anomalies
wt_mod_signal = wt_mod_anom.mean(dim='run_index').rename('signal') #get year-to-year signal time series based on anomalies
wt_mod_noise = wt_mod_anom.std(dim='run_index').rename('noise') #get year-to-year noise time series based on anomalies
wt_mod_snr = (wt_mod_signal / wt_mod_noise).rename('snr') #year-to-year ensemble signal to noise ratio
wt_mod_signal_tmean = (np.abs(wt_mod_signal)).mean(dim='time') #get temporal mean of the absolute signal time series
wt_mod_noise_tmean = wt_mod_noise.mean(dim='time') #get temporal mean noise

#get ensemble mean and std for the raw (non-anomaly) LWT counts
if anom == 'no':
    wt_mod_mean = wt_mod.mean(dim='run_index').rename('ensemble_mean_count') #year-to-year ensemble mean values
    #wt_mod_std = wt_mod.std(dim='run_index').rename('ensemble_std_count') #year-to-year ensemble standard deviation values
elif anom == 'yes':
    print('Upon user request, the correlation coefficients and RPC score will be calculated upon anomaly counts !')
    wt_obs = wt_obs_anom #overwrite raw observed counts with anomalies
    wt_mod = wt_mod_anom #overwrite raw model counts with anomalies
    wt_mod_mean = wt_mod.mean(dim='run_index').rename('ensemble_mean_count_based_on_anomalies') #year-to-year ensemble mean values
else:
    raise Exception('ERROR: check entry for <anom> input parameter !')

#replicate observations to have the same dimensions as the <wt_mod_mean> model array containing the raw year-to-year ensemble-mean counts
wt_obs = xr.concat([wt_obs] * len(experiment), dim='experiment') #replicate observarions along <experiment> dimension
wt_obs = xr.DataArray(wt_obs.transpose('lwt','season','experiment','time','lon','lat').values,coords=wt_mod_mean.coords,dims=wt_mod_mean.dims,name='replicated_observed_timeseries')

wt_mod_mean_mem = xr.concat([wt_mod_mean] * len(wt_mod.run_index), dim='run_index') #replicate ensemble-mean value along <run_index> dimension
wt_mod_mean_mem = xr.DataArray(wt_mod_mean_mem.transpose('lwt','season','experiment','run_index','time','lon','lat').values,coords=wt_mod.coords,dims=wt_mod.dims,name='replicated_ensemble_mean_timeseries')
# wt_mod_mean_mem = wt_mod_mean_mem.assign_coords(run_index=wt_mod.run_index) #add coordinate <run_index>
# wt_mod_mean_mem = wt_mod_mean_mem.transpose('lwt','season','experiment','run_index','time','lon','lat') #reorder dimensions

#calculate correlation measures between the ensemble-mean (i.e. the signal) and observations
pearson_r = xs.pearson_r(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('pearson_r')
pearson_pval = xs.pearson_r_p_value(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('pearson_pval')
pearson_pval_effn = xs.pearson_r_eff_p_value(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('pearson_pval_effn')
spearman_r = xs.spearman_r(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('spearman_r')
spearman_pval = xs.spearman_r_p_value(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('spearman_pval')
spearman_pval_effn = xs.spearman_r_eff_p_value(wt_obs,wt_mod_mean,dim='time',skipna=True).rename('spearman_pval_effn')

## RPC calculated with ensemble mean and total variance following Eade et al. 2014, see equation 1 in doi:10.1002/2014GL061146
var_sig = wt_mod_mean.var(dim='time') #calculate the variance of the ensemble-mean time series
var_tot = wt_mod.var(dim='time').mean(dim='run_index') #calculate the mean of the individual members' variance
rpc_var = pearson_r / np.sqrt(var_sig / var_tot) #calculate the RPC based on signal / total model variance
# pc_mod = np.sqrt(var_sig/var_tot) #modelled predictable componenent, i.e. potential predictability

## RPC according to equation 5 in Scaife et al. 2018, https://doi.org/10.1038/s41612-018-0038-4
pc_mod = xs.pearson_r(wt_mod,wt_mod_mean_mem,dim='time',skipna=True).mean(dim='run_index').rename('pearson_r')
rpc = (pearson_r / pc_mod).rename('rpc_based_on_rho') #calculate the RPC based on the average correlation coefficient between the signal and individual members
rpc_expvar = np.sqrt(pearson_r**2 / pc_mod**2).rename('rpc_based_on_explained_var') #calculate the RPC based on explained variance

##RPC according to equation 1 in Kumar et al. 2014, Monthly Weather review
var_sig_mem = xr.concat([var_sig] * len(wt_mod.run_index), dim='run_index') #replicate ensemble-mean value along <run_index> dimension
var_sig_mem = var_sig_mem.assign_coords(run_index=wt_mod.run_index) #add coordinate <run_index>
var_sig_mem = var_sig_mem.transpose('lwt','season','experiment','run_index','lon','lat') #reorder dimensions to match <pc_mod_kumar>
var_sig_mem = xr.DataArray(var_sig_mem.values,coords=[var_sig_mem.lwt,var_sig_mem.season,var_sig_mem.experiment,var_sig_mem.run_index,var_sig_mem.lon,var_sig_mem.lat],dims=var_sig_mem.dims,name='signal_variance')
var_mem = wt_mod.var(dim='time').rename('member_variance') #member variance
ac_perfect = (wt_mod - wt_mod_mean).mean(dim='time') / (var_sig_mem*var_mem).mean(dim='run_index')
ac = (wt_mod_mean - wt_obs).mean(dim='time') / (var_sig*wt_obs.var(dim='time'))
rpc_kumar = ac / ac_perfect

#test the significane of the difference between the observed and modelled predictable components, interpreted as two correlation coefficients
# Fisher z transformation
z1 = 0.5 * np.log((1 + pearson_r) / (1 - pearson_r))
z2 = 0.5 * np.log((1 + pc_mod) / (1 - pc_mod))
# Number of samples
n1 = len(wt_mod.time)
n2 = len(wt_mod.time)
# Standard errors
se1 = 1 / np.sqrt(n1 - 3)
se2 = 1 / np.sqrt(n2 - 3)
# Standard error of the difference between correlations
sed = np.sqrt(se1**2 + se2**2)
# Calculate z-score for the difference
z_diff = (z1 - z2) / sed
# Compute p-value
p_value = 2 * (1 - norm.cdf(np.abs(z_diff)))
rpc_sigind = p_value < 1-test_level/100
rpc_sigind = xr.DataArray(rpc_sigind,coords=rpc.coords,dims=rpc.dims,name='sig_diff_pc')

# #get critical values for the RPC following the supplementary material available from Eade et al. 2014, doi:10.1002/2014GL061146, available at https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2F2014GL061146&file=Eade_070714_AUXILIARY_MATERIAL.pdf
# rpc = get_rpc(wt_mod,wt_mod_mean,pearson_r)
# ntime = len(wt_mod.time)
# nmem = len(wt_mod.run_index)
# #orig_run_index = wt_mod.run_index.values #for use with all members
# orig_run_index = wt_mod.run_index.values[0:nmem-exclude_members] #for use with a sub-selection of members
# orig_time = wt_mod.time.values
# rpc_rand = np.zeros((repetitions,rpc.shape[0],rpc.shape[1],rpc.shape[2],rpc.shape[3],rpc.shape[4]))
# for rr in list(np.arange(repetitions)):
    # #reshuffle
    # rand1 = np.random.randint(0,ntime,ntime)
    # #rand2 = np.random.randint(0,nmem,nmem) #reshuffle all ensemble members
    # rand2 = np.random.randint(0,nmem-exclude_members,nmem-exclude_members) #reshuffle of fraction of ensemble members only
    # wt_mod_rand = wt_mod[:,:,:,:,rand1,:,:] #reshuffle the time instances (years)
    # wt_mod_rand = wt_mod_rand[:,:,:,rand2,:,:,:] #reshuffle the members
    # wt_mod_rand.time.values[:] = orig_time #nominally get back original <time> dimension in ascending order for xskillsore to work below
    # wt_mod_rand.run_index.values[:] = orig_run_index #nominally get back original <run_index> dimension in ascending order for xskillsore to work below
    
    # wt_mod_mean_rand = wt_mod_rand.mean(dim='run_index') #calculate the ensemble-mean of the reshuffled model array
    # #get random scores
    # pearson_r_rand = xs.pearson_r(wt_obs,wt_mod_mean_rand,dim='time',skipna=True).rename('pearson_r')
    # rpc_rand_step = get_rpc(wt_mod_rand,wt_mod_mean_rand,pearson_r_rand) #use reshuffled model array to obtain the total variance
    # #rpc_rand_step = get_rpc(wt_mod,wt_mod_mean_rand,pearson_r_rand) #use original model array to obtain the total variance 
    
    # #assign
    # rpc_rand[rr,:,:,:,:] = rpc_rand_step
    # #and clean
    # rpc_rand_step.close()
    # wt_mod_rand.close()
    # wt_mod_mean_rand.close()
    # del(rpc_rand_step,wt_mod_rand,wt_mod_mean_rand,rand1,rand2)
# #get the critical values and significance True False used below for plotting above the rpc countour
# rpc_critval = np.nanpercentile(rpc_rand,[test_level/2,100-test_level/2],axis=0) #quantiles must be given as percentage value (0-100), e.g. 2.5 and 97.5 for a test-level of 5% and a two-sided configuration
# # sig_below_one_ind = np.where((rpc_critval[0,:,:,:,:,:] < 1) & (rpc_critval[1,:,:,:,:,:] < 1))
# # sig_greater_one_ind = np.where((rpc_critval[0,:,:,:,:,:] > 1) & (rpc_critval[1,:,:,:,:,:] > 1))
# sig_below_one_ind = (rpc_critval[0,:,:,:,:,:] < 1) & (rpc_critval[1,:,:,:,:,:] < 1)
# sig_greater_one_ind = (rpc_critval[0,:,:,:,:,:] > 1) & (rpc_critval[1,:,:,:,:,:] > 1)
# rpc_sigind = np.zeros(sig_below_one_ind.shape,dtype='bool')
# rpc_sigind[sig_below_one_ind] = True
# rpc_sigind[sig_greater_one_ind] = True
# #rpc_sigind = (rpc < rpc_critval[0,:,:,:,:,:]) | (rpc > rpc_critval[1,:,:,:,:,:]) #check whether the actual rpc is smaller than the lower percentile or larger then the upper percentile, in which case it signifcantly differes from 0
# rpc_sigind = xr.DataArray(rpc_sigind,coords=rpc.coords,dims=rpc.dims,name='sig_diff_from_1')

#and plot the results
for lwt in np.arange(len(tarwts_name)):
    for sea in np.arange(len(seasons)):
        #obs_seas = wt_obs.sel(lwt=tarwts_name[lwt],season=seasons[sea])
        for en in np.arange(len(experiment_out)):
            #get correlation measueres for the specific LWT, season and experiment
            spearman_r_step = spearman_r.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            spearman_pval_step = spearman_pval.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            
            #plot nh and sh separately
            spearman_r_nh = spearman_r_step.isel(lat=spearman_r_step.lat >= 0)
            spearman_r_sh = spearman_r_step.isel(lat=spearman_r_step.lat < 0)
            spearman_pval_nh = spearman_pval_step.isel(lat=spearman_pval_step.lat >= 0)
            spearman_pval_sh = spearman_pval_step.isel(lat=spearman_pval_step.lat < 0)
            sig_ind_nh = (spearman_pval_nh.values < 0.05) & (spearman_r_nh.values > 0)
            sig_ind_sh = (spearman_pval_sh.values < 0.05) & (spearman_r_sh.values > 0)
            xx_nh, yy_nh = np.meshgrid(spearman_r_nh.lon.values,spearman_r_nh.lat.values)
            xx_sh, yy_sh = np.meshgrid(spearman_r_sh.lon.values,spearman_r_sh.lat.values)
            
            title = 'Rank corr. coeff., dtr'+detrending+', '+tarwts_name[lwt]+', '+seasons[sea]+', '+experiment_out[en]+' vs '+obs.upper()+', '+str(taryears[0])+', '+str(taryears[1])
            savedir = figs+'/'+experiment[en]+'/global/'+tarwts_name[lwt]+'/correlation/maps'
            if os.path.isdir(savedir) != True:
                os.makedirs(savedir)
            halfres = (lat.values[1]-lat.values[0])/2
            cbarlabel_spear = 'Rank correlation coefficient'
            savename_nh = savedir+'/spearman_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            savename_sh = savedir+'/spearman_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            get_map_lowfreq_var(np.transpose(spearman_r_nh.values),xx_nh,yy_nh,-1,1,dpival,title,savename_nh,halfres,colormap,titlesize,cbarlabel_spear,map_proj_nh,agree_ind=np.transpose(sig_ind_nh),origpoint=None)
            get_map_lowfreq_var(np.transpose(spearman_r_sh.values),xx_sh,yy_sh,-1,1,dpival,title,savename_sh,halfres,colormap,titlesize,cbarlabel_spear,map_proj_sh,agree_ind=np.transpose(sig_ind_sh),origpoint=None)
            del(sig_ind_nh,sig_ind_sh)
            
            #get RPC measures for the specific LWT, season and experiment
            rpc_step = rpc.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            rpc_sigind_step = rpc_sigind.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            
            #plot nh and sh separately
            rpc_nh = rpc_step.isel(lat=rpc_step.lat >= 0)
            rpc_sh = rpc_step.isel(lat=rpc_step.lat < 0)
            sig_ind_nh = rpc_sigind_step.isel(lat=rpc_sigind_step.lat >= 0).values
            sig_ind_sh = rpc_sigind_step.isel(lat=rpc_sigind_step.lat < 0).values
            #sig_ind_nh = (rpc_nh.values > 1)
            #sig_ind_sh = (rpc_sh.values > 1)
            xx_nh, yy_nh = np.meshgrid(rpc_nh.lon.values,rpc_nh.lat.values)
            xx_sh, yy_sh = np.meshgrid(rpc_sh.lon.values,rpc_sh.lat.values)
            
            title = 'RPC, dtr'+detrending+', '+tarwts_name[lwt]+', '+seasons[sea]+', '+experiment_out[en]+' vs '+obs.upper()+', '+str(taryears[0])+', '+str(taryears[1])
            savedir = figs+'/'+experiment[en]+'/global/'+tarwts_name[lwt]+'/rpc/maps'
            if os.path.isdir(savedir) != True:
                os.makedirs(savedir)
            halfres = (lat.values[1]-lat.values[0])/2
            cbarlabel_rpc = 'Ratio of Predictable Components'
            savename_nh = savedir+'/rpc_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            savename_sh = savedir+'/rpc_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            get_map_lowfreq_var(np.transpose(rpc_nh.values),xx_nh,yy_nh,0,2,dpival,title,savename_nh,halfres,colormap,titlesize,cbarlabel_rpc,map_proj_nh,agree_ind=np.transpose(sig_ind_nh),origpoint=None)
            get_map_lowfreq_var(np.transpose(rpc_sh.values),xx_sh,yy_sh,0,2,dpival,title,savename_sh,halfres,colormap,titlesize,cbarlabel_rpc,map_proj_sh,agree_ind=np.transpose(sig_ind_sh),origpoint=None)
            del(sig_ind_nh,sig_ind_sh)           
            
            #get temporal mean snr ratios for the specific LWT, season and experiment from dcppa and historical and then calculate the ratio dcppa / historical
            snr_dcppa_step = np.abs(wt_mod_snr.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])).mean(dim='time')
            snr_hist_step = np.abs(wt_mod_snr.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment='historical')).mean(dim='time')
            snr_ratio_step = snr_dcppa_step / snr_hist_step # calculate the ratio of the temporal-mean signal-to-noise ratios
            #snr_ratio_step = snr_dcppa_step.mean(dim='time') / snr_hist_step.mean(dim='time') # calculate the ratio of the temporal-mean signal-to-noise ratios
            #snr_ratio_step = snr_ratio_step.mean(dim='time') #calculate the temporal mean of the ratio of signal-to-noise ratios
            snr_critval = 2 / np.sqrt(len(wt_mod.run_index)-1) #obtain the criticval value for a significant SNR
            
            #plot nh and sh separately
            snr_ratio_nh = snr_ratio_step.isel(lat=snr_ratio_step.lat >= 0)
            snr_ratio_sh = snr_ratio_step.isel(lat=snr_ratio_step.lat < 0)
            snr_dcppa_nh = snr_dcppa_step.isel(lat=snr_ratio_step.lat >= 0)
            snr_dcppa_sh = snr_dcppa_step.isel(lat=snr_ratio_step.lat < 0)
            sig_ind_nh = (snr_ratio_nh.values > 1) & (snr_dcppa_nh.values > snr_critval)
            sig_ind_sh = (snr_ratio_sh.values > 1) & (snr_dcppa_sh.values > snr_critval)
            xx_nh, yy_nh = np.meshgrid(snr_ratio_nh.lon.values,snr_ratio_nh.lat.values)
            xx_sh, yy_sh = np.meshgrid(snr_ratio_sh.lon.values,snr_ratio_sh.lat.values)
            
            title = 'Ratio of temporal mean SNR for '+experiment_out[en]+' / SNR for historical, dtr'+detrending+', '+tarwts_name[lwt]+', '+seasons[sea]+', '+str(taryears[0])+', '+str(taryears[1])
            savedir = figs+'/'+experiment[en]+'/global/'+tarwts_name[lwt]+'/snr_ratio/maps'
            if os.path.isdir(savedir) != True:
                os.makedirs(savedir)
            halfres = (lat.values[1]-lat.values[0])/2
            cbarlabel_snr = 'SNR dcppA / SNR hist.'
            savename_nh = savedir+'/snr_ratio_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_historical_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            savename_sh = savedir+'/snr_ratio_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_historical_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            get_map_lowfreq_var(np.transpose(snr_ratio_nh.values),xx_nh,yy_nh,0,2,dpival,title,savename_nh,halfres,colormap,titlesize,cbarlabel_snr,map_proj_nh,agree_ind=np.transpose(sig_ind_nh),origpoint=None)
            get_map_lowfreq_var(np.transpose(snr_ratio_sh.values),xx_sh,yy_sh,0,2,dpival,title,savename_sh,halfres,colormap,titlesize,cbarlabel_snr,map_proj_sh,agree_ind=np.transpose(sig_ind_sh),origpoint=None)
            del(sig_ind_nh,sig_ind_sh)
            
            #map temporal mean noise term for the specific LWT, season and experiment, the temporal mean has been already calculated above, outside the current loop
            noise_step = wt_mod_noise_tmean.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            title = 'Temporal-mean noise term for '+experiment_out[en]+' dtr'+detrending+', '+tarwts_name[lwt]+', '+seasons[sea]+', '+str(taryears[0])+', '+str(taryears[1])
            cbarlabel_noise = 'Temporal-mean noise'
            savedir = figs+'/'+experiment[en]+'/global/'+tarwts_name[lwt]+'/noise/maps'
            savename_nh = savedir+'/noise_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            savename_sh = savedir+'/noise_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            map_polar_single_field(noise_step,title,savename_nh,savename_sh,wt_mod_noise_tmean.min(),wt_mod_noise_tmean.max(),dpival,colormap,titlesize,cbarlabel_noise)
            
            #map temporal mean signal term for the specific LWT, season and experiment, the temporal mean has been already calculated above, outside the current loop
            signal_step = wt_mod_signal_tmean.sel(lwt=tarwts_name[lwt],season=seasons[sea],experiment=experiment_out[en])
            cbarlabel_signal = 'Temporal-mean absolute signal'
            title = 'Temporal-mean signal term for '+experiment_out[en]+' dtr'+detrending+', '+tarwts_name[lwt]+', '+seasons[sea]+', '+str(taryears[0])+', '+str(taryears[1])
            savedir = figs+'/'+experiment[en]+'/global/'+tarwts_name[lwt]+'/signal/maps'
            savename_nh = savedir+'/signal_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            savename_sh = savedir+'/signal_anom_'+anom+'_dtr_'+detrending+'_'+tarwts_name[lwt]+'_'+seasons[sea]+'_'+experiment_out[en]+'_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
            map_polar_single_field(signal_step,title,savename_nh,savename_sh,wt_mod_signal_tmean.min(),wt_mod_signal_tmean.max(),dpival,colormap,titlesize,cbarlabel_signal)
            
            # #plot nh and sh separately
            # noise_nh = noise_step.isel(lat=noise_step.lat >= 0)
            # noise_sh = noise_step.isel(lat=noise_step.lat < 0)
            # sig_ind_nh = None
            # sig_ind_sh = None
            # xx_nh, yy_nh = np.meshgrid(noise_nh.lon.values,noise_nh.lat.values)
            # xx_sh, yy_sh = np.meshgrid(noise_sh.lon.values,noise_sh.lat.values)
            
            #clean up temporary xarray objects
            snr_dcppa_step.close()
            snr_hist_step.close()
            snr_ratio_step.close()
            spearman_pval_step.close()
            rpc_sigind_step.close()
            signal_step.close()
            noise_step.close()
            spearman_pval_nh.close()
            spearman_pval_sh.close()
            spearman_r_nh.close()
            spearman_r_sh.close()
            spearman_r_step.close()
            snr_ratio_nh.close()
            snr_ratio_sh.close()
            snr_dcppa_nh.close()
            snr_dcppa_sh.close()

#make bar plots
#get hemispheric-wise results
spearman_r_nh = spearman_r.isel(lat=spearman_r.lat >= 0)
spearman_r_sh = spearman_r.isel(lat=spearman_r.lat < 0)
spearman_pval_nh = spearman_pval.isel(lat=spearman_pval.lat >= 0)
spearman_pval_sh = spearman_pval.isel(lat=spearman_pval.lat < 0)
boxplot_xticks = np.arange(len(tarwts_name))
#get critical values for the correlation coefficient with a two-tailed test
df = len(wt_mod.time)- 2 #Degrees of freedom (assuming n > 30, approximately)
alpha = (100 - test_level)/100
critical_value = t.ppf(1 - alpha / 2, df) #calculate the critical value

#loop through seasons
for sea in np.arange(len(seasons)):
    #loop through experiments
    for en in np.arange(len(experiment_out)):
        savedir = figs+'/'+experiment[en]+'/global/summary/correlation/boxplots'
        wtlabel2save = str(tarwts_name).replace('[','').replace(']','').replace("'",'').replace(', ','_')
        if os.path.isdir(savedir) != True:
            os.makedirs(savedir)
        #get dcppa and historical (i.e. reference) results and stack lon and lat to new <gridpoint> dimension
        spearman_r_nh_step = spearman_r_nh.sel(season=seasons[sea],experiment=experiment_out[en]).stack(gridpoints=['lon','lat'])
        spearman_r_sh_step = spearman_r_sh.sel(season=seasons[sea],experiment=experiment_out[en]).stack(gridpoints=['lon','lat'])

        savename_nh = savedir+'/boxplot_spearman_dtr_'+detrending+'_'+wtlabel2save+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_nh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
        fig = plt.figure()
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[0,0],linestyle='dotted',color='black',zorder=1)
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[rho_critval,rho_critval],linestyle='dotted',color='red',zorder=1)
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[rho_critval*-1,rho_critval*-1],linestyle='dotted',color='red',zorder=1)
        sns.boxplot(data=spearman_r_nh_step.transpose('gridpoints','lwt'))
        plt.xticks(ticks=boxplot_xticks,labels=tarwts_name)
        plt.xlabel('Lamb Weather Type')
        plt.ylim([np.abs(spearman_r).max()*-1,np.abs(spearman_r).max()]) #y axis is centred around 0
        plt.ylabel(cbarlabel_spear)
        #plt.grid(axis='y',linestyle='dotted')
        plt.savefig(savename_nh,dpi=dpival)
        plt.close('all')
        del(fig)

        savename_sh = savedir+'/boxplot_spearman_dtr_'+detrending+'_'+wtlabel2save+'_'+seasons[sea]+'_'+experiment_out[en]+'_vs_'+obs+'_sh_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
        fig = plt.figure()
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[0,0],linestyle='dotted',color='black')
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[rho_critval,rho_critval],linestyle='dotted',color='red')
        plt.plot([-0.5,boxplot_xticks[-1]+0.5],[rho_critval*-1,rho_critval*-1],linestyle='dotted',color='red')
        sns.boxplot(data=spearman_r_sh_step.transpose('gridpoints','lwt'))
        plt.xticks(ticks=boxplot_xticks,labels=tarwts_name)
        plt.xlabel('Lamb Weather Type')
        plt.ylim([np.abs(spearman_r).max()*-1,np.abs(spearman_r).max()]) #y axis is centred around 0
        plt.ylabel(cbarlabel_spear)
        #plt.grid(axis='y',linestyle='dotted')
        plt.savefig(savename_sh,dpi=dpival)
        plt.close('all')
        del(fig)
        
        #clean temporary xr data arrays
        spearman_r_nh_step.close()
        spearman_r_sh_step.close()

#clean remaining xarray objects
wt_mod_noise_tmean.close()
wt_obs.close()
wt_obs_tmean.close()
wt_obs_anom.close()
wt_mod_mean_mem.close()
pearson_r.close()
pearson_pval.close()
pearson_pval_effn.close()
spearman_r.close()
spearman_pval.close()
spearman_pval_effn.close()
#expvar_mod.close()
#expvar_mod_mean.close()
#corr_mod.close()
#corr_mod_mean.close()
rpc.close()
rpc_sigind.close()
wt_mod_signal_tmean.close()
wt_obs.close()
wt_mod_tmean.close()
wt_mod_anom.close()
wt_mod_signal.close()
wt_mod_noise.close()
wt_mod_mean.close()
#wt_mod_std.close()
wt_mod_snr.close()

print('INFO: skill_maps_from_mon_counts.py has run successfully!')