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
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('analysis_functions.py').read())
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)

#set input parameter
ensemble = ['era5','cera20c','ec_earth3','ec_earth3'] #cera20c or mpi_esm_1_2_hr or ec_earth3
ensemble_color = ['orange','black','grey','blue']
ensemble_linestyle = ['dashed','dashed','solid','dotted']
experiment = ['20c','20c','dcppA','historical'] #historical, amip, piControl, 20c or dcppA
experiment_out = ['20c_era5','20c_cera20c','dcppA','historical'] #used to distinct several 20c experiments in the xr data array created by this script. Is important when calculating correlations.
lead_time = [1,5,10] #lead time or forecast year or the dcppA LWT data

## this is the configuration for use without ERA5
# taryears_obs = [[1961,2010],[1965,2010],[1970,2010]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
# taryears_hist = [[1961,2019],[1965,2023],[1970,2028]]
# taryears_obs = [[1901,2010],[1901,2010],[1901,2010]]

## this is the configuration for use with ERA5
#taryears_obs1 = [[1940,2022],[1940,2022],[1940,2022]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
taryears_obs1 = [[1961,2022],[1961,2022],[1961,2022]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
taryears_obs2 = [[1961,2010],[1961,2010],[1961,2010]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
#taryears_obs2 = [[1901,2010],[1901,2010],[1901,2010]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
taryears_hist = [[1961,2028],[1961,2028],[1961,2028]]
taryears_dcppa = [[1961,2019],[1965,2023],[1970,2028]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA

#city = ['Bergen','Paris','Prague','Barcelona'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
#city = ['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
city = ['Wellington','SantiagoDC'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest

reference_period = [1970,2014] # "from_data" or list containing the start and end years
seasons = ['ONDJFM','JJA'] #list of seasons to be considered: year, DJF, MAM, JJA or SON, ONDJFM or AMJJAS
tarwts = [7,15,24] #[7,15,24] westerlies, [5,13,22] southerlies, [9,17,26] northerly, 15 = pure directional west
center_wrt = 'memberwise_mean' # ensemble_mean or memberwise_mean; centering w.r.t. to ensemble (or overall) mean value or member-wise temporal mean value prior to calculating signal-to-noise

figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #base path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
meanperiod = 10 #running-mean period in years
std_critval = 1.28 #1 = 68%, 1.28 = 80 %, 2 = 95%; standard deviation used to define the critical value above or below which the signal-to-noise ratio is assumed to be significant.
rho_ref = '20c_era5' #20c_era5 or 20c_cera20c; reference observational dataset used to calculate correlations. Must be included in the <experiment_out> input parameter defined above

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
yearly_units = '%' # count, % (relative frequency) or z-score; unit of the yearly LWT counts

#visualization options
axes_config = 'equal' # equal or individual; if set to equal, the axes limits are equal for all considered lead-times in a given city; if set to individual, the axes are optimized for each specific lead-time in that city
plot_sig_stn_only = 'no' #plot only significant signal-to-noise ratios in pcolor format, yes or no
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures
colormap = 'viridis'
edgecolor = 'black' #if set to any colour, e.g. black, cell boundaries are depicted in the pcolor plots generated by this script.

#auxiliary variables to be used in future versions of the script; currently not used
aggreg = 'year' #unlike map_lowfreq_var.py, this script currently only works for aggreg = 'year', other aggregation levels will be implemented in the future
anom = 'no' #not relevant here since yearly counts are considered, i.e. the annual cycle is not present in the time series, is kept for future versions of the script

##execute ###############################################################################################
#check correct usage of the script
if len(city) < 2:
    raise Exception('ERROR: a minimum of two cities must be defined in the <city> input parameter !')

#taryears = np.stack((taryears_obs,taryears_dcppa,taryears_hist))
#taryears = xr.DataArray(taryears,coords=[experiment,lead_time,np.arange(len(taryears_obs[0]))],dims=['experiment','lead_time','years'], name='temporal_coverage')

taryears = np.stack((taryears_obs1,taryears_obs2,taryears_dcppa,taryears_hist))
taryears = xr.DataArray(taryears,coords=[experiment,lead_time,np.arange(len(taryears_obs1[0]))],dims=['experiment','lead_time','years'], name='temporal_coverage')

#set the reference period
if reference_period == 'from_data':
    ref_period = [taryears[:,:,0].max().values,taryears[:,:,1].min().values] #first and last year of the reference period common to all lead-times and experiments that is used for anomaly calculation below
else:
    ref_period = reference_period
print('The reference period used for anomaly calculation is '+str(ref_period))

if aggreg != 'year': #check for correct usage of the script
    raise Exception("Bad call of the script: This script currently only works for yearly LWT counts, i.e. aggreg = 'year' !)")

wtnames = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

study_years = np.arange(np.array(taryears).min(),np.array(taryears).max()+1,1) #numpy array of years limited by the lower and upper end of considered years
t_startindex = int(meanperiod/2) #start index of the period to be plotted along the x axis
t_endindex = int(len(study_years)-(meanperiod/2-1)) #end index of the period to be plotted along the x axis

#init output arrays of the ensemble x city loop series
#for 3d arrays ensemble x city x study_years
stn_all = np.zeros((len(seasons),len(lead_time),len(ensemble),len(city),len(study_years)))
signal_all = np.copy(stn_all)
noise_all = np.copy(stn_all)
runmeans_all = np.copy(stn_all)

#CALCULATIONS part of the script
##loop through all seasons, sea
for sea in np.arange(len(seasons)):
    seaslabel = seasons[sea]
    ##loop throught the lead-times, lt
    for lt in np.arange(len(lead_time)):
        #loop throught the ensembles, en
        for en in np.arange(len(ensemble)):
            
            #define the start and end years requested for this ensemble/experiment and lead-timefssea
            start_year_step = taryears[en][lt][0].values
            end_year_step = taryears[en][lt][1].values
            
            #get ensemble configuration as defined in analysis_functions,py, see get_ensemble_config() function therein
            model,mrun,model_label,tarhours = get_ensemble_config(ensemble[en],experiment[en])
            if experiment[en] == 'dcppA':
                mrun_dcppA = mrun
            elif experiment[en] == 'historical':
                mrun_historical = mrun
            elif experiment[en] == '20c':
                mrun_20c = mrun
            else:
                raise Exception('ERROR: unknown entry for <experiment[en]> !')
            
            #init numpy arrays needing ensemble size in <mrun> variable
            if sea == 0 and lt == 0 and en == 0:
                runmeans_i_all = np.zeros((len(seasons),len(lead_time),len(ensemble),len(mrun),len(city),len(study_years)))
                wt_agg_tmean_all = np.zeros((len(seasons),len(lead_time),len(ensemble),len(mrun),len(city))) #temporal LWT frequency average for each lead-time, ensemble, run and city, calculated upon running average values
                wt_agg_tstd_all = np.zeros((len(seasons),len(lead_time),len(ensemble),len(mrun),len(city))) #temporal LWT frequency standard deviation for each lead-time, ensemble, run and city, calculated upon running average values
            
            print('INFO: get annual and decadal mean time series as well as signal-to-noise ratios for '+aggreg+' LWT counts, '+ensemble[en]+' with '+str(len(mrun[en]))+' runs, '+str(seasons[sea])+' months, '+str(taryears[en][lt])+' years, '+str(tarhours)+' hours. Output LWT frequency units are: '+yearly_units+'.')
            wt_agg = np.zeros((len(model),len(list(range(start_year_step,end_year_step+1)))))
            
            #loop trought each city, cc
            for cc in np.arange(len(city)):
                print('INFO: obtaining results for '+city[cc])
                #get target location
                tarlat,tarlon = get_location(city[cc])
                if tarlon < 0:
                    tarlon = 360+tarlon # convert target longitude into 0 - 360 degrees format
                #detect the hemisphere
                if tarlat > 0:
                    hemis = 'nh'
                elif tarlat <= 0:
                    hemis = 'sh'
                else:
                    raise Exception('ERROR: unknown entry for <tarlat>!')
                
                for mm in list(range(len(model))):
                    #get metadata for this GCM, incluing start and end year label of the LWT input files to be loaded below
                    if experiment[en] in ('amip','dcppA','historical','piControl'):
                        runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm]) #check whether historical GCM configurations agree with those used in DCPPA ! 
                        file_taryears, timestep = get_target_period(model[mm],experiment[en],cmip_f=cmip,lead_time_f=lead_time[lt])
                    elif experiment[en] == '20c':
                        print('get_historical_metadata.py is not called for experiment = '+experiment[en]+'...')
                        file_taryears, timestep = get_target_period(model[mm],experiment[en])
                    else:
                        raise Exception('ERROR: unknown entry in <experiment> input parameter')
                    
                    file_startyear = file_taryears[0]
                    file_endyear = file_taryears[1]
                    
                    store_wt = store_wt_orig+'/'+timestep+'/'+experiment[en]+'/'+hemis
                    if experiment[en] == 'dcppA':
                        wt_file = store_wt+'/wtseries_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_'+hemis+'_'+str(lead_time[lt])+'y_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the LWT catalogues
                    elif experiment[en] in ('historical', 'amip', '20c'):
                        wt_file = store_wt+'/wtseries_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_'+hemis+'_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the LWT catalogues
                    else:
                        raise Excpetion('ERROR: Unknown entry for <experiment> input paramter !')

                    #load the LWT time series for the centerlat and target years obtained above
                    wt = xr.open_dataset(wt_file)
                    
                    #filter out the data for the requested season
                    if seasons[sea] in ('MAM','JJA','SON'):
                        wt = wt.sel(time=(wt['time'].dt.season == seasons[sea]))
                        print('Processing '+seasons[sea]+' season...')
                    elif seasons[sea] in ('DJF'):
                        wt = wt.isel(time = np.isin(wt['time'].dt.month,[1,2,12]))
                        print('Processing '+seasons[sea]+' season...')    
                    elif seasons[sea] in ('ONDJFM'):
                        wt = wt.isel(time = np.isin(wt['time'].dt.month,[1,2,3,10,11,12]))
                        print('Processing '+seasons[sea]+' season...')                    
                    elif seasons[sea] in ('AMJJAS'):
                        wt = wt.isel(time = np.isin(wt['time'].dt.month,[4,5,6,7,8,9]))
                        print('Processing '+seasons[sea]+' season...')
                    elif seasons[sea] == 'year':
                        print('For season[sea] = '+seasons[sea]+', the entire calendar year will be considered.')
                    else:
                        raise Exception('ERROR: Unknown entry for '+seasons[sea]+' !!')
                        
                    # #add one year for DJF values
                    # if seasons[sea] == 'DJF':
                        # print('Adding one year to December datetime64 index values for '+seasons[sea]+'...')
                        # dates_pd = pd.DatetimeIndex(wt.time.values)
                        # dates_pd[dates_pd.month==12][:] = dates_pd[dates_pd.month==12]+pd.DateOffset(years=1)
                        # wt.time.values[:] = dates_pd
                    
                    #get the gridbox nearest to tarlon and tarlat
                    lon_center = wt.lon.sel(lon=tarlon,method='nearest').values
                    lat_center = wt.lat.sel(lat=tarlat,method='nearest').values        
                    wt_center = wt.sel(lon=lon_center,lat=lat_center,method='nearest')
                    
                    #select requested time period (years, season and hours)
                    dates_wt = pd.DatetimeIndex(wt_center.time.values)
                    if model[mm] == 'ec_earth3' and mrun[mm] == 'r1i1p1f1' and experiment[en] == 'dcppA': #add exception for EC-Earth3, r1i1p1f1, dcppA experiment which provides mean instead of instantaneous SLP values that are stored in the centre of the time interval
                        print('WARNING: For '+experiment[en]+', '+model[mm]+' and '+mrun[mm]+' 6-hourly mean SLP values centred on the time interval are provided ! For consistence with the other members, 3 hours are added to <tarhours> and the search for common timesteps is repeated ! ')
                        tarhours_alt = list(np.array(tarhours)+3)
                        year_ind_wt = np.where((dates_wt.year >= start_year_step) & (dates_wt.year <= end_year_step) & (np.isin(dates_wt.hour,tarhours_alt)))[0]
                    else:
                        year_ind_wt = np.where((dates_wt.year >= start_year_step) & (dates_wt.year <= end_year_step) & (np.isin(dates_wt.hour,tarhours)))[0]

                    wt_center = wt_center.isel(time=year_ind_wt)

                    #modify origional LWT time series containing 27 types to a binary absence (0) - occurrence (1) time series of the requested types only
                    bin_array = np.zeros(wt_center.wtseries.shape)
                    tarwt_ind = np.where(wt_center.wtseries.isin(tarwts))
                    bin_array[tarwt_ind] = 1
                    arr_tarwts = xr.DataArray(data=bin_array,coords=[pd.DatetimeIndex(wt_center.time)],dims='time',name='wtseries')
                    
                    #get time series with yearly target WT counts
                    if aggreg == 'year':                
                        #process accumulation period
                        if seasons[sea] in ('MAM','JJA','SON'):
                            hours_per_year = arr_tarwts.groupby('time.year').count() #returns the exact number of time instances per year taking into account leap-years
                            wt_agg_step = arr_tarwts.groupby('time.year').sum('time')
                        elif seasons[sea] in ('DJF'):
                            hours_per_year = (arr_tarwts.time.dt.year == arr_tarwts.time.dt.year[0]).sum() # a single value specifying the number or time instances of the first year on record   
                            arr_tarwts = arr_tarwts.rolling(time=int(hours_per_year)).sum() #rolling sum                        
                            end_hour = np.sort(np.unique(arr_tarwts.time.dt.hour.values))[-1] #get the last hour of the accumulation period
                            wt_agg_step = arr_tarwts.isel(time=np.where((arr_tarwts.time.dt.month == 2) & (arr_tarwts.time.dt.day == 28) & (arr_tarwts.time.dt.hour == end_hour))[0]) #get the accumulated values ending in March
                            wt_agg_step = wt_agg_step.groupby('time.year').sum('time',skipna=False) #does not sum anything but changes the time coordinate to "year"
                        elif seasons[sea] in ('ONDJFM'):
                            hours_per_year = (arr_tarwts.time.dt.year == arr_tarwts.time.dt.year[0]).sum() # a single value specifying the number or time instances of the first year on record   
                            arr_tarwts = arr_tarwts.rolling(time=int(hours_per_year)).sum() #rolling sum                        
                            end_hour = np.sort(np.unique(arr_tarwts.time.dt.hour.values))[-1] #get the last hour of the accumulation period
                            wt_agg_step = arr_tarwts.isel(time=np.where((arr_tarwts.time.dt.month == 3) & (arr_tarwts.time.dt.day == 31) & (arr_tarwts.time.dt.hour == end_hour))[0]) #get the accumulated values ending in March
                            wt_agg_step = wt_agg_step.groupby('time.year').sum('time',skipna=False) #does not sum anything but changes the time coordinate to "year"
                        elif seasons[sea] in ('AMJJAS'):
                            hours_per_year = (arr_tarwts.time.dt.year == arr_tarwts.time.dt.year[0]).sum() # a single value specifying the number or time instances of the first year on record   
                            arr_tarwts = arr_tarwts.rolling(time=int(hours_per_year)).sum() #rolling sum                        
                            end_hour = np.sort(np.unique(arr_tarwts.time.dt.hour.values))[-1] #get the last hour of the accumulation period
                            wt_agg_step = arr_tarwts.isel(time=np.where((arr_tarwts.time.dt.month == 9) & (arr_tarwts.time.dt.day == 30) & (arr_tarwts.time.dt.hour == end_hour))[0]) #get the accumulated values ending in March
                            wt_agg_step = wt_agg_step.groupby('time.year').sum('time',skipna=False) #does not sum anything but changes the time coordinate to "year"
                        else:
                            raise Excpetion('Error: check entry for <season[sea]> !')
                        
                        #process output units
                        if yearly_units == 'count':
                            ylabel_ts = 'Yearly occurrence frequency (count)'
                        elif yearly_units == '%':
                            ylabel_ts = 'Yearly relative occurrence frequency (%)'
                            #wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                            #wt_agg_step = wt_agg_step / hours_per_year *100
                            wt_agg_step = wt_agg_step / hours_per_year *100
                        elif yearly_units == 'z-score':
                            ylabel_ts = 'z-score'
                            #wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                            wt_agg_step = z_transform(wt_agg_step / hours_per_year *100)
                        else:
                            raise Exception('ERROR: unknown entry for <yearly_units>!')       
                    
                    ntime = wt_agg_step.values.shape[0]
                    wt_agg[mm,:] = wt_agg_step.values #fill into the numpy array <wt_agg> for further processing
                    
                #set output directories
                comparison_dir = figs+'/'+model[mm]+'/'+wtlabel.replace(" ","_") #directory to save summary figures used for comparing different experiments, cities, lead_times etc.
                timeseries_dir = figs+'/'+model[mm]+'/'+experiment[en]+'/local/'+city[cc]+'/'+aggreg+'/timeseries' #experiment-specific output figure directory
                #create target directory if missing
                if os.path.isdir(timeseries_dir) != True:
                    os.makedirs(timeseries_dir)
                if os.path.isdir(comparison_dir) != True:
                    os.makedirs(comparison_dir)
                
                #get running mean of yearly ensemble mean counts; then calculate and plot the signal-to-noise ratio as well as its critival value
                years = wt_agg_step.year.values
                wt_agg = xr.DataArray(wt_agg,coords=[np.arange(wt_agg.shape[0]),years],dims=['member','time'],name='wtfreq')
                year_ind_anom = np.where((wt_agg.time >= ref_period[0]) & (wt_agg.time <= ref_period[1]))[0] #get indices of the years within the reference period
                wt_agg_tmean = wt_agg.isel(time=year_ind_anom).mean(dim='time') #tmean = temporal mean for each member, calculated upon reference years only
                wt_agg_tstd = wt_agg.isel(time=year_ind_anom).std(dim='time') #tmean = temporal mean for each member, calculated upon reference years only
                if center_wrt == 'memberwise_mean':
                    anom_i = wt_agg - np.tile(wt_agg_tmean,(len(years),1)).transpose()
                elif center_wrt == 'ensemble_mean':
                    anom_i = wt_agg - wt_agg_tmean.mean()
                
                #caclulate running mean anomalies and signal-to-noise ratios thereon
                runanom_i = anom_i.rolling(time=meanperiod,center=True,min_periods=None).mean()
                #runsignal = runanom_i.mean(dim='member').rename('signal') #ensemble signal for each temporal mean period
                runsignal = wt_agg.mean(dim='member').rename('signal') #ensemble signal for each temporal mean period
                runnoise = runanom_i.std(dim='member').rename('noise') #ensemble standard deviation for each temporal mean period
                runstn = np.abs(runsignal / runnoise).rename('signal-to-noise') #signal-to-noise ration for each temporal mean period
                critval_stn = runstn.copy() #get a new xarray data array from an existing one
                critval_stn[:] = std_critval / np.sqrt(len(model)-1) #fill this new xarray data array with the critival value for a significant signal-to-noise ratio, as defined by the first equation in https://doi.org/10.1007/s00382-010-0977-x (Deser et al. 2012, Climate Dynamics)
                
                #caclulated raw / non transformed running LWT frequencies
                means = wt_agg.mean(axis=0)
                runmeans = means.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values of the yearly ensemble mean values
                runmeans_i = wt_agg.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values for each member
                
                #plot signal-to-noise time series individually for each experiment and city, and compare with critical value
                min_occ = runstn.copy() #get a new xarray data array from an existing one
                min_occ[:] = wt_agg.min().values #overall minimum yearly WT frequency of all runs, used for plotting purpose below            
                fig = plt.figure()
                critval_stn.plot()
                runstn.plot()
                #runnoise.plot()
                if experiment[en] == 'dcppA':
                    savename_stn = timeseries_dir+'/STN_'+seasons[sea]+'_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
                else:
                    savename_stn = timeseries_dir+'/STN_'+seasons[sea]+'_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
                plt.savefig(savename_stn,dpi=dpival)
                plt.close('all')
                del(fig)
                
                #plot the yearly and running temporall mean time series for each individual member and for the ensemble mean
                fig = plt.figure()
                years_mat = np.transpose(np.tile(years,[len(model),1]))
                plt.plot(years_mat,np.transpose(wt_agg.values),linewidth=0.5,color='grey',linestyle='dotted')
                plt.plot(years,runmeans,linewidth=2,color='black')
                plt.plot(years_mat,np.transpose(runmeans_i.values),linewidth=1)
                if any(runstn > critval_stn): #if the signal is significant for any temporal mean value, then depict this with a marker
                    #runmeans[runstn > critval_stn].plot(linestyle='None',marker='o',markersize=6,color='orange',zorder=0) #plot significant ensemble mean values (signals)
                    plt.plot(years[runstn > critval_stn],min_occ[runstn > critval_stn],linestyle='None',marker='D',markersize=6,color='red') #plot significant ensemble mean values (signals)
                
                #set x axis configuration
                plt.xlabel('year')
                if axes_config == 'equal':
                    #set x axis configuration, the same range of years is plotted for each lead-time (or forecast year), considering the longest period without NaNs available from the joint data arrays of all considered lead-times
                    plt.xticks(ticks=study_years[t_startindex:t_endindex][9::10],labels=study_years[t_startindex:t_endindex][9::10])
                    plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
                    #y limits correspond to the minimum and maximum value of the running-mean relative frequency of the requested LWT in the specific city
                    plt.ylim([runmeans_i.min().values,runmeans_i.max().values])
                elif axes_config == 'individual':
                    #the range of the x-axis is optimized for each individual lead time
                    plt.xticks(ticks=years[9::10],labels=years[9::10])
                else:
                    raise Exception('ERROR: check entry for <axes_config> input parameter !')
                
                #set y axis configuration
                plt.ylabel(ylabel_ts)
                plt.ylim([wt_agg.min(),wt_agg.max()])
                
                plt.title('LWT '+wtlabel+' '+city[cc]+' '+model_label+' '+str(len(mrun))+' members '+str(start_year_step)+'-'+str(end_year_step))
                text_x = np.percentile(years,30) # x coordinate of text inlet
                text_y = wt_agg.values.max() - (wt_agg.values.max() - wt_agg.values.min())/25 # y coordinate of text inlet
                #plt.text(text_x,text_y, '$\sigma$ / $\mu$ = '+str(np.round(np.nanstd(runmeans)/np.nanmean(runmeans),3)),size=8) #plot standard deviation of running ensemble mean time series as indicator of forced response
                plt.text(text_x,text_y, 'temporal mean and minimum $\sigma$ = '+str(np.round(np.nanmean(runnoise),3))+' and '+str(np.round(np.nanmin(runnoise),3)),size=8) #plot temporal mean of the running standard deviation of the decadal mean LWT frequency anomalies from each member
                if experiment[en] == 'dcppA':
                    savename_ts = timeseries_dir+'/timeseries_'+seasons[sea]+'_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
                else:
                    savename_ts = timeseries_dir+'/timeseries_'+seasons[sea]+'_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
                plt.savefig(savename_ts,dpi=dpival)
                plt.close('all')
                del(fig)
                wt_center.close()
                wt.close()
                
                #save the signal, noise and signal-to-noise ratios for this ensemble and city
                nanseries_stn = np.zeros(len(study_years)) #generate nan time series covering the entire time period
                nanseries_stn[:] = np.nan
                nanseries_signal = np.copy(nanseries_stn)
                nanseries_noise = np.copy(nanseries_stn)
                nanseries_runmeans = np.copy(nanseries_stn)
                nanseries_runmeans_i = np.tile(nanseries_stn,(len(mrun),1))
                #find boolean True for years or this ensemble contained in study_years and paste the results into the extended array
                fill_bool = np.isin(study_years,runstn.time.values)
                nanseries_stn[fill_bool] = runstn.values
                nanseries_signal[fill_bool] = runsignal.values
                nanseries_noise[fill_bool] = runnoise.values
                nanseries_runmeans[fill_bool] = runmeans.values
                for mem in np.arange(len(mrun)):
                    nanseries_runmeans_i[mem,fill_bool] = runmeans_i[mem,:].values
                
                #assign
                stn_all[sea,lt,en,cc,:] = nanseries_stn
                signal_all[sea,lt,en,cc,:] = nanseries_signal
                noise_all[sea,lt,en,cc,:] = nanseries_noise
                runmeans_all[sea,lt,en,cc,:] = nanseries_runmeans
                runmeans_i_all[sea,lt,en,:,cc,:] = nanseries_runmeans_i
                wt_agg_tmean_all[sea,lt,en,:,cc] = wt_agg_tmean
                wt_agg_tstd_all[sea,lt,en,:,cc] = wt_agg_tstd
                #assign mrun, will be overwritten in each loop through lead_time

#convert output numpy arrays to xarray data array
stn_all = xr.DataArray(stn_all,coords=[seasons,lead_time,experiment_out,city,study_years],dims=['season','lead_time','experiment','city','time'],name='signal-to-noise')
signal_all = xr.DataArray(signal_all,coords=[seasons,lead_time,experiment_out,city,study_years],dims=['season','lead_time','experiment','city','time'],name='signal')
noise_all = xr.DataArray(noise_all,coords=[seasons,lead_time,experiment_out,city,study_years],dims=['season','lead_time','experiment','city','time'],name='noise')
runmeans_all = xr.DataArray(runmeans_all,coords=[seasons,lead_time,experiment_out,city,study_years],dims=['season','lead_time','experiment','city','time'],name='running_ensemble_mean')
runmeans_i_all = xr.DataArray(runmeans_i_all,coords=[seasons,lead_time,experiment_out,np.arange(len(mrun)),city,study_years],dims=['season','lead_time','experiment','member','city','time'],name='running_member_mean')
wt_agg_tmean_all = xr.DataArray(wt_agg_tmean_all,coords=[seasons,lead_time,experiment_out,np.arange(len(mrun)),city],dims=['season','lead_time','experiment','member','city'],name='member_mean')
wt_agg_tstd_all = xr.DataArray(wt_agg_tstd_all,coords=[seasons,lead_time,experiment_out,np.arange(len(mrun)),city],dims=['season','lead_time','experiment','member','city'],name='member_std')
stn_all.experiment.attrs['ensemble'] = ensemble
signal_all.experiment.attrs['ensemble'] = ensemble
noise_all.experiment.attrs['ensemble'] = ensemble
runmeans_all.experiment.attrs['ensemble'] = ensemble
runmeans_i_all.experiment.attrs['ensemble'] = ensemble
wt_agg_tmean_all.experiment.attrs['ensemble'] = ensemble
wt_agg_tstd_all.experiment.attrs['ensemble'] = ensemble

#PLOTTING part of the script
#loop through the seasons
for sea in np.arange(len(seasons)):
    #for signal, noise and signal-to-noise ratios, the reanalysis is excluded, the focus is put on model experiments
    
    ## use .isel() for the latter 3 xr arrays because in former versions of the script the .sel() did not work due to repetition of the the obs = '20c' experiment within the array
    # stn_model = stn_all.isel(experiment = np.isin(stn_all.experiment,['dcppA','historical'])).sel(season=seasons[sea])
    # noise_model = noise_all.isel(experiment = np.isin(noise_all.experiment,['dcppA','historical'])).sel(season=seasons[sea])
    # wt_agg_tmean_model = wt_agg_tmean_all.isel(experiment = np.isin(wt_agg_tmean_all.experiment,['dcppA','historical'])).sel(season=seasons[sea])
    
    stn_model = stn_all.sel(season=seasons[sea], experiment = ['dcppA','historical'])
    noise_model = noise_all.sel(season=seasons[sea], experiment = ['dcppA','historical'])
    wt_agg_tmean_model = wt_agg_tmean_all.sel(season=seasons[sea], experiment = ['dcppA','historical'])
    
    stn_model_max = stn_model.max().values #maximum signal-to-noise ratio across all model experiments and lead-times
    noise_model_min = noise_model.min().values
    noise_model_max = noise_model.max().values #currently not in use yet, maximum noise / standard deviation across all model experiments and lead-times1
    
    #init output arrays
    rho1_all = np.zeros((len(lead_time),len(city)))
    rho2_all = np.copy(rho1_all)
    rho3_all = np.copy(rho2_all)
    ##loop throught the lead-times
    for lt in np.arange(len(lead_time)):
        #plot city-scale results
        critvals_study_period = np.tile(critval_stn[0].values,len(study_years))
        exp_label = str(experiment).replace('[','').replace(']','').replace("'","").replace(', ','_')
        for cc in np.arange(len(city)):
            fig = plt.figure()
            plt.plot(stn_all.time.values,critvals_study_period)
            #for exp in np.arange(len(experiment_out)):
            for exp in ['historical','dcppA']:
                stn_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment=exp,city=city[cc]).plot()
            plt.ylim(0,stn_all.sel(experiment=['historical','dcppA']).max().values) #the maximum for all GCM experiments i.e. CERA-20C and ERA5 (in this script is a dummy ensemble with ten equal members and stn = inf anywhere) are excluded
            plt.title(wtlabel.replace(" ","_")+'_'+seasons[sea]+'_FY'+str(lead_time[lt])+', '+city[cc])
            plt.legend(['critval']+['historical','dcppA'])
            #savename_stn_city = figs+'/'+model[mm]+'/timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            savename_stn_city = comparison_dir+'/stn_stdcritval_'+seasons[sea]+'_'+str(std_critval)+'_timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            plt.savefig(savename_stn_city,dpi=dpival)
            plt.close('all')
            del(fig)
            
            #plot the yearly and running temporal mean time series for each individual member and for the ensemble mean
            fig = plt.figure()
            study_years_mat = np.transpose(np.tile(study_years,[len(model),1]))
            for en in np.arange(len(ensemble)):
                plt.plot(study_years,runmeans_all[sea,lt,en,cc,:].values,linewidth=2,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
                plt.plot(study_years_mat,np.transpose(runmeans_i_all[sea,lt,en,:,cc,:].values),linewidth=0.5,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
                #plt.plot(years_mat,np.transpose(runmeans_i.values),linewidth=1)
            
            #set x and y axis configuration
            if axes_config == 'equal':
                #the same range of years is plotted for each lead-time (or forecast year), considering the longest period without NaNs
                plt.xticks(ticks=study_years[t_startindex:t_endindex][9::10],labels=study_years[t_startindex:t_endindex][9::10])
                plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
                
                # #y limits correspond to the minimum and maximum value of the running-mean relative frequency of the requested LWT in the specific city
                # plt.ylim([runmeans_i_all.sel(city=city[cc]).min().values,runmeans_i_all.sel(city=city[cc]).max().values])
                
                #equal y axis limit for all experiments of the same city
                plt.ylim([runmeans_i_all.sel(season=seasons[sea],city=city[cc]).min().values,runmeans_i_all.sel(season=seasons[sea],city=city[cc]).max().values])
            elif axes_config == 'individual':
                print('The x-axis in the time-series plot is optimized for the '+seasons[sea]+' season and lead-time = '+str(lead_time[lt])+' years.')
            else:
                raise Exception('ERROR: unknown entry for <axes_config> input parameter !')
            
            #plot corrleation coefficients in the titles
            rho1 = xr.corr(runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment='dcppA',city=city[cc]),runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment='historical',city=city[cc])) #calculate the correlation between the two running ensemble decadal mean time series
            rho2 = xr.corr(runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment='dcppA',city=city[cc]),runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment=rho_ref,city=city[cc]))
            rho3 = xr.corr(runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment='historical',city=city[cc]),runmeans_all.sel(season=seasons[sea],lead_time=lead_time[lt],experiment=rho_ref,city=city[cc]))
            rho1_all[lt,cc] = rho1
            rho2_all[lt,cc] = rho2
            rho3_all[lt,cc] = rho3
            plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', FY'+str(lead_time[lt])+', '+city[cc]+': r(dcppA-hist) = '+str(rho1.round(2).values)+', r(dcppA-obs) = '+str(rho2.round(2).values)+', r(hist-obs) = '+str(rho3.round(2).values), size = 8)
            
            #save the figure
            savename_ts_all = comparison_dir+'/timeseries_'+seasons[sea]+'_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            plt.savefig(savename_ts_all,dpi=dpival)
            plt.close('all')
            del(fig)

        #plot summary city-scale results, difference in signal-to-noise ratio dcppA minus historical in pcolor format
        fig = plt.figure()
        stn_diff = stn_all.sel(season=seasons[sea],lead_time=lead_time[lt], experiment='dcppA') - stn_all.sel(season=seasons[sea],lead_time=lead_time[lt], experiment='historical')
        stn_diff.plot(edgecolors=edgecolor) #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
        #xlim2pcolor = [study_years[~np.isnan(stn_diff[0,:])][0]-0.5,study_years[~np.isnan(stn_diff[0,:])][-1]+0.5]
        #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
        plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
        plt.ylim(-0.5,len(city)-0.5)
        plt.title(wtlabel.replace(" ","_")+', SNR dcppA '+seasons[sea]+' FY'+str(lead_time[lt])+' minus SNR historical')
        savename_diff_stn = comparison_dir+'/pcolor_diffstn_'+seasons[sea]+'_'+exp_label+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_diff_stn,dpi=dpival)
        plt.close('all')
        del(fig)
        
        #ab hier !
        #plot signal-to-noise ratio separately for dcppA and historical    
        if plot_sig_stn_only == 'yes':
            print('WARNING: Only significant signal-to-noise ratios are plotted in pcolor format!')
            stn_model = stn_model.where(stn_model > critval_stn[0])
        elif  plot_sig_stn_only == 'no':
            stn_model_sig = stn_model.where(stn_model > critval_stn[0])
        else:
            raise Exception('ERROR: unknown entry for <stn_model_sig> input parameter !')

        for exp in stn_model.experiment.values:
            #get significant SNR for the given lead-time and experiment
            stn_model_sig_step = stn_model_sig.sel(lead_time=lead_time[lt],experiment=exp)
            fig = plt.figure()
            stn_model.sel(lead_time=lead_time[lt], experiment=exp).plot(vmin=0,vmax=stn_model_max,edgecolors=edgecolor,cmap=colormap) #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
            
            if plot_sig_stn_only == 'no':
                #loop through the (city x time period) matrix and mark those running time periods where the signal-to-noise ratio is significant with a dot 
                city_pos = np.arange(len(stn_model_sig_step.city)) #auxiliary variable used to find the y - coordinate of the city in the pcolor plotted below
                for ii in np.arange(stn_model_sig_step.shape[0]):
                    for jj in np.arange(stn_model_sig_step.shape[1]):
                        if ~np.isnan(stn_model_sig_step[ii,jj]):
                            plt.plot(stn_model_sig_step.time[jj],stn_model_sig_step.city[ii],linestyle='none',color='red',marker = 'o', markersize=4)
                            #plt.plot(stn_model_sig_step.time[jj].values,city_pos[ii],linestyle='none',color='red',marker = 'o', markersize=4)
            
            #xlim2pcolor = [study_years[~np.isnan(stn_diff[0,:])][0]-0.5,study_years[~np.isnan(stn_diff[0,:])][-1]+0.5]    
            #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
            plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
            plt.ylim(-0.5,len(city)-0.5)
            #set title and savename
            if exp == 'dcppA':
                savename_dcppa_stn = comparison_dir+'/pcolor_stn_stdcritval_'+seasons[sea]+'_'+str(std_critval)+'_'+exp+'_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
                plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', '+exp+', FY'+str(lead_time[lt])+', SNR')
            else:
                savename_dcppa_stn = comparison_dir+'/pcolor_stn_stdcritval_'+seasons[sea]+'_'+str(std_critval)+'_'+exp+'_paired_with_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
                plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', '+exp+' paired with dcppA FY'+str(lead_time[lt])+', SNR')
            plt.savefig(savename_dcppa_stn,dpi=dpival)
            plt.close('all')
            del(fig)

        #Pcolor, difference between the standard deviation from dcppA with specific forecast year and historical for each running mean period
        fig = plt.figure()
        noise_diff = noise_all.sel(season=seasons[sea],lead_time=lead_time[lt], experiment='dcppA') - noise_all.sel(season=seasons[sea],lead_time = lead_time[lt], experiment='historical')
        noise_diff.plot(edgecolors=edgecolor) #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
        #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
        plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
        plt.ylim(-0.5,len(city)-0.5)
        plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', dcppa FY'+str(lead_time[lt])+' minus historical')
        savename_diff_noise = comparison_dir+'/pcolor_diffstd_'+seasons[sea]+'_dcppa_FY'+str(lead_time[lt])+'y_minus_hist_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_diff_noise,dpi=dpival)
        plt.close('all')
        del(fig)
        
        #Pcolor, standard deviation of dcppA with specific forecast year for each running mean period
        fig = plt.figure()
        noise_all.sel(season=seasons[sea],lead_time=lead_time[lt], experiment='dcppA').plot(vmin=noise_model_min,vmax=noise_model_max,cmap=colormap,edgecolors=edgecolor)
        #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
        plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
        plt.ylim(-0.5,len(city)-0.5)
        plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', dcppa FY'+str(lead_time[lt]) + ' standard deviation')
        savename_dcppa_noise = comparison_dir+'/pcolor_std_'+seasons[sea]+'_dcppa_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_dcppa_noise,dpi=dpival)
        plt.close('all')
        del(fig)
        
        #Pcolor, standard deviation of historical experiment for each running mean period
        fig = plt.figure()
        noise_all.sel(season=seasons[sea],lead_time=lead_time[lt], experiment='historical').plot(vmin=noise_model_min,vmax=noise_model_max,cmap=colormap,edgecolors=edgecolor)
        #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
        plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
        plt.ylim(-0.5,len(city)-0.5)
        plt.title(wtlabel.replace(" ","_")+', '+seasons[sea]+', historical paired with dcppa FY'+str(lead_time[lt]) + ' standard deviation')
        #plt.gca().set_aspect('equal')
        savename_dcppa_noise = comparison_dir+'/pcolor_std_'+seasons[sea]+'_historical_paired_with_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_dcppa_noise,dpi=dpival)
        plt.close('all')
        del(fig)

    #Pcolor, temporal mean values for all members and forecast years of the dcppA experiment as well as of the respective period from the historical experiment, one pcolor per city
    #concat dcppa and hist experiments
    wt_agg_tmean_concat = xr.concat([wt_agg_tmean_all.sel(experiment='dcppA'), wt_agg_tmean_all.sel(experiment='historical')],dim='lead_time')
    wt_agg_tmean_concat.lead_time.values[:] = np.arange(len(wt_agg_tmean_concat.lead_time)) ##modify lead-time dimension to increase monotonically
    wt_agg_tmean_std = wt_agg_tmean_concat.std(dim='member') #calculates standard deviation of the memberwise mean values for dcppa and hist in all cities
    for cc in np.arange(len(city)):
        minval_tmean = wt_agg_tmean_model.sel(city=city[cc]).min().values
        maxval_tmean = wt_agg_tmean_model.sel(city=city[cc]).max().values
        
        #concat
        fig = plt.figure()
        wt_agg_tmean_concat.sel(season=seasons[sea],city=city[cc]).plot(vmin=minval_tmean,vmax=maxval_tmean,cmap=colormap,edgecolors=edgecolor)
        std_step = wt_agg_tmean_std.sel(season=seasons[sea],city=city[cc])
        yticks_tmean = wt_agg_tmean_concat.lead_time[0:len(lead_time)+1]
        ylabels_fy = list(wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA').lead_time.values.astype(str))+['hist']
        #ylabels_tmean = list(wt_agg_tmean_all.sel(experiment='dcppA').lead_time.values.astype(str))+['hist']
        ylabels_tmean = [ylabels_fy[ii]+' '+str(std_step[ii].round(2).values)[1:] for ii in np.arange(len(ylabels_fy))] #implicit loop to construct ylabels plus standard deviation of the temporal mean values from the individual members
        plt.yticks(ticks = yticks_tmean, labels = ylabels_tmean)
        plt.ylim(yticks_tmean[0]-0.5,yticks_tmean[-1]+0.5)
        plt.ylabel('forecast year and std of the mean')
        plt.title(wtlabel.replace(" ","_")+' in '+city[cc]+', dcppA plus hist, member-wise clim. mean freq.')
        savename_concat_tmean = comparison_dir+'/pcolor_tmean_'+seasons[sea]+'_concat_dcppa_historical_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_concat_tmean,dpi=dpival)
        plt.close('all')
        del(fig)

#plot figures involving all seasons
for sea in np.arange(len(seasons)):
    fig = plt.figure()
    for cc in np.arange(len(city)):
        #get statistics to be plotted
        hist_min = wt_agg_tmean_all.sel(season=seasons[sea],experiment='historical',city=city[cc],lead_time=1).min().values
        hist_max = wt_agg_tmean_all.sel(season=seasons[sea],experiment='historical',city=city[cc],lead_time=1).max().values
        hist_mean = wt_agg_tmean_all.sel(season=seasons[sea],experiment='historical',city=city[cc],lead_time=1).mean().values
        
        fy1_min = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=1).min().values
        fy1_max = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=1).max().values
        fy1_mean = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=1).mean().values

        fy5_min = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=5).min().values
        fy5_max = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=5).max().values
        fy5_mean = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=5).mean().values
        
        fy10_min = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=10).min().values
        fy10_max = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=10).max().values
        fy10_mean = wt_agg_tmean_all.sel(season=seasons[sea],experiment='dcppA',city=city[cc],lead_time=10).mean().values
        
        obs_mean = wt_agg_tmean_all.sel(season=seasons[sea],experiment=rho_ref,city=city[cc],lead_time=1).mean().values
        
        #and plot them
        plt.plot([hist_min,hist_max],[cc, cc],color='blue')
        plt.plot(hist_mean,cc,color='blue',linestyle='None',marker='D')        
        plt.plot([fy1_min,fy1_max],[cc+0.3, cc+0.3],color='green')
        plt.plot(fy1_mean,cc+0.3,color='green',linestyle='None',marker='D')        
        plt.plot([fy5_min,fy5_max],[cc+0.2, cc+0.2],color='red')
        plt.plot(fy5_mean,cc+0.2,color='red',linestyle='None',marker='D')        
        plt.plot([fy10_min,fy10_max],[cc+0.1, cc+0.1],color='black')
        plt.plot(fy10_mean,cc+0.1,color='black',linestyle='None',marker='D')        
        plt.plot(obs_mean,cc+0.4,color='orange',linestyle='None',marker='D')
    plt.xlabel('Relative frequency (%)')
    yticks = np.arange(len(city))+0.2
    plt.yticks(ticks=yticks,labels=city)
    plt.legend(['HIST spread','HIST mean','FY1 spread','FY1 mean','FY5 spread','FY5 mean','FY10 spread','FY10 mean','ERA5 mean'])
    savename_barplots = comparison_dir+'/barplot_histspread_dcppAmean_allcities_'+seasons[sea]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(ref_period[0])+'_'+str(ref_period[-1])+'.'+outformat
    plt.savefig(savename_barplots,dpi=dpival)
    plt.close('all')
    
    #plot the raw and standardized dcppA (forecast year 1) and historical signal, as well as the observed time-series for each city
    for sea in np.arange(len(seasons)):
        for cc in np.arange(len(city)):
            #calculate Pearson correlation coefficient between ERA5 and dcppA ensemble-mean time series for the raw data
            rho_dcppa_era5 = xr.corr(signal_all.sel(experiment='20c_era5',season=seasons[sea],city=city[cc],lead_time=1),signal_all.sel(experiment='dcppA',season=seasons[sea],city=city[cc],lead_time=1))
            rho_cera20c_era5 = xr.corr(signal_all.sel(experiment='20c_era5',season=seasons[sea],city=city[cc],lead_time=1),signal_all.sel(experiment='20c_cera20c',season=seasons[sea],city=city[cc],lead_time=1))
            #plot raw values
            fig = plt.figure()
            signal_all.sel(experiment='20c_era5',season=seasons[sea],city=city[cc],lead_time=1).plot(color='black',label='ERA5')
            signal_all.sel(experiment='20c_cera20c',season=seasons[sea],city=city[cc],lead_time=1).plot(color='grey',linestyle='dotted',label='CERA-20C')
            signal_all.sel(experiment='dcppA',season=seasons[sea],city=city[cc],lead_time=1).plot(color='red',label='dcppA')
            #signal_all.sel(experiment='historical',season=seasons[sea],city=city[cc],lead_time=1).plot(color='blue')
            
            plt.title(wtlabel+' in '+seasons[sea]+': '+str(np.round(rho_dcppa_era5.values,2))+' / '+str(np.round(rho_cera20c_era5.values,2)))
            plt.legend()
            savename_ts_signal_raw = comparison_dir+'/timeseries_raw_'+seasons[sea]+'_'+wtlabel.replace(" ","_")+'_'+str(study_years[0])+'_'+str(study_years[-1])+'_'+city[cc]+'.'+outformat
            plt.savefig(savename_ts_signal_raw,dpi=dpival)
            plt.close('all')

            #plot standardized anomalies
            fig = plt.figure()
            z_transform(signal_all.sel(experiment='20c_era5',season=seasons[sea],city=city[cc],lead_time=1)).plot(color='black',label='ERA5')
            z_transform(signal_all.sel(experiment='20c_cera20c',season=seasons[sea],city=city[cc],lead_time=1)).plot(color='grey',linestyle='dotted',label='CERA-20C')
            z_transform(signal_all.sel(experiment='dcppA',season=seasons[sea],city=city[cc],lead_time=1)).plot(color='red',label='dcppA')
            #z_transform(signal_all.sel(experiment='historical',season=seasons[sea],city=city[cc],lead_time=1)).plot(color='blue')
            
            plt.title(wtlabel+' in '+seasons[sea]+': '+str(np.round(rho_dcppa_era5.values,2))+' / '+str(np.round(rho_cera20c_era5.values,2)))
            plt.legend()
            savename_ts_signal_std = comparison_dir+'/timeseries_std_'+seasons[sea]+'_'+wtlabel.replace(" ","_")+'_'+str(study_years[0])+'_'+str(study_years[-1])+'_'+city[cc]+'.'+outformat
            plt.savefig(savename_ts_signal_std,dpi=dpival)
            plt.close('all')

print('INFO: signal2noise_local_extended_season.py has run successfully!')
