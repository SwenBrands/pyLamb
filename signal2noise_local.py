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
exec(open('analysis_functions.py').read())
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)

#set input parameter
ensemble = ['cera20c','ec_earth3','ec_earth3'] #cera20c or mpi_esm_1_2_hr or ec_earth3
ensemble_color = ['black','grey','blue']
ensemble_linestyle = ['dashed','solid','dotted']
experiment = ['20c','dcppA','historical'] #historical, amip, piControl, 20c or dcppA
#lead_time = [1,5,10] #lead time or forecast year or the dcppA LWT data
lead_time = [1,5,10] #lead time or forecast year or the dcppA LWT data

# taryears_obs = [[1961,2010],[1965,2010],[1970,2010]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
# taryears_hist = [[1961,2019],[1965,2023],[1970,2028]]
taryears_obs = [[1961,2010],[1961,2010],[1961,2010]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA
taryears_hist = [[1961,2028],[1961,2028],[1961,2028]]
taryears_dcppa = [[1961,2019],[1965,2023],[1970,2028]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA

#city = ['Barcelona','Bergen','Paris','Prague'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
#city = ['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
city = ['Barcelona','Bergen'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest

tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #target months
tarwts = [1] #[5,13,22] direcciones sur, [9,17,26] direcciones norte, 15 = purely directional west
center_wrt = 'memberwise_mean' # ensemble_mean or memberwise_mean; centering w.r.t. to ensemble (or overall) mean value or member-wise temporal mean value prior to calculating signal-to-noise

figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
meanperiod = 10 #running-mean period in years

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
yearly_units = '%' # count, % (relative frequency) or z-score; unit of the yearly LWT counts

#visualization options
axes_config = 'equal' # equal or individual; if set to equal, the axes limits are equal for all considered lead-times in a given city; if set to individual, the axes are optimized for each specific lead-time in that city
plot_sig_stn_only = 'no' #plot only significant signal-to-noise ratios in pcolor format, yes or no
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures

#auxiliary variables to be used in future versions of the script; currently not used
aggreg = 'year' #unlike map_lowfreq_var.py, this script currently only works for aggreg = 'year', other aggregation levels will be implemented in the future
anom = 'no' #not relevant here since yearly counts are considered, i.e. the annual cycle is not present in the time series, is kept for future versions of the script

#execute ###############################################################################################
taryears = np.stack((taryears_obs,taryears_dcppa,taryears_hist))
taryears = xr.DataArray(taryears,coords=[experiment,lead_time,np.arange(len(taryears_obs[0]))],dims=['experiment','lead_time','years'], name='temporal_coverage')
ref_period = [taryears[:,:,0].max().values,taryears[:,:,1].min().values] #first and last year of the reference period common to all lead-times and experiments that is used for anomaly calculation below

if aggreg != 'year': #check for correct usage of the script
    raise Exception("Bad call of the script: This script currently only works for yearly LWT counts, i.e. aggreg = 'year' !)")

seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','')
if seaslabel == '123456789101112':
    seaslabel = 'yearly'

wtnames = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

study_years = np.arange(np.array(taryears).min(),np.array(taryears).max()+1,1) #numpy array of years limited by the lower and upper end of considered years
t_startindex = int(meanperiod/2) #start index of the period to be plotted along the x axis
t_endindex = int(len(study_years)-(meanperiod/2-1)) #end index of the period to be plotted along the x axis

#init output arrays of the ensemble x city loop series
#for 3d arrays ensemble x city x study_years
stn_all = np.zeros((len(lead_time),len(ensemble),len(city),len(study_years)))
signal_all = np.copy(stn_all)
noise_all = np.copy(stn_all)
runmeans_all = np.copy(stn_all)

##loop throught the lead-times
for lt in np.arange(len(lead_time)):
    #loop throught the ensembles
    for en in np.arange(len(ensemble)):
        
        #define the start and end years requested for this ensemble/experiment and lead-time
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
        if lt == 0 and en == 0:
            runmeans_i_all = np.zeros((len(lead_time),len(ensemble),len(mrun),len(city),len(study_years)))
            wt_agg_tmean_all = np.zeros((len(lead_time),len(ensemble),len(mrun),len(city))) #temporal LWT frequency average for each lead-time, ensemble, run and city, calculated upon running average values
        
        print('INFO: get annual and decadal mean time series as well as signal-to-noise ratios for '+aggreg+' LWT counts, '+ensemble[en]+' with '+str(len(mrun[en]))+' runs, '+str(tarmonths)+' months, '+str(taryears[en][lt])+' years, '+str(tarhours)+' hours. Output LWT frequency units are: '+yearly_units+'.')
        wt_agg = np.zeros((len(model),len(list(range(start_year_step,end_year_step+1)))))
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
                
                #get the gridbox nearest to tarlon and tarlat
                lon_center = wt.lon.sel(lon=tarlon,method='nearest').values
                lat_center = wt.lat.sel(lat=tarlat,method='nearest').values        
                wt_center = wt.sel(lon=lon_center,lat=lat_center,method='nearest')
                
                #select requested time period (years and hours)
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
                    if yearly_units == 'count':
                        ylabel_ts = 'Yearly occurrence frequency (count)'
                        wt_agg_step = arr_tarwts.groupby('time.year').sum('time')
                    elif yearly_units == '%':
                        ylabel_ts = 'Yearly relative occurrence frequency (%)'
                        hours_per_year = arr_tarwts.groupby('time.year').count()
                        wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                        wt_agg_step = wt_agg_step / hours_per_year *100
                    elif yearly_units == 'z-score':
                        ylabel_ts = 'z-score'
                        hours_per_year = arr_tarwts.groupby('time.year').count()
                        wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                        wt_agg_step = z_transform(wt_agg_step / hours_per_year *100)
                    else:
                        raise Exception('ERROR: unknown entry for <yearly_units>!')       
                ntime = wt_agg_step.values.shape[0]
                wt_agg[mm,:] = wt_agg_step.values #fill into the numpy array <wt_agg> for further processing
                
            #Plot year-to-year WT counts for each ensemble member of CERA-20C and the <meanperiod>-year rolling temporal average of the yearly ensemble mean counts
            timeseries_dir = figs+'/'+model[mm]+'/'+experiment[en]+'/local/'+city[cc]+'/'+aggreg+'/timeseries'
            #create target directory if missing
            if os.path.isdir(timeseries_dir) != True:
                os.makedirs(timeseries_dir)
            
            #get running mean of yearly ensemble mean counts; then calculate and plot the signal-to-noise ratio as well as its critival value
            years = wt_agg_step.year.values
            wt_agg = xr.DataArray(wt_agg,coords=[np.arange(wt_agg.shape[0]),years],dims=['member','time'],name='wtfreq')
            year_ind_anom = np.where((wt_agg.time >= ref_period[0]) & (wt_agg.time <= ref_period[1]))[0] #get indices of the years within the reference period
            wt_agg_tmean = wt_agg.isel(time=year_ind_anom).mean(dim='time') #tmean = temporal mean for each member, calculated upon reference years only
            np.tile(wt_agg_tmean,(len(years),1)).transpose()
            if center_wrt == 'memberwise_mean':
                anom_i = wt_agg - np.tile(wt_agg_tmean,(len(years),1)).transpose()
            elif center_wrt == 'ensemble_mean':
                anom_i = wt_agg - wt_agg_tmean.mean()
            
            #caclulate running mean anomalies and signal-to-noise ratios thereon
            runanom_i = anom_i.rolling(time=meanperiod,center=True,min_periods=None).mean()
            runsignal = runanom_i.mean(dim='member').rename('signal') #ensemble signal for each temporal mean period
            runnoise = runanom_i.std(dim='member').rename('noise') #ensemble standard deviation for each temporal mean period
            runstn = np.abs(runsignal / runnoise).rename('signal-to-noise') #signal-to-noise ration for each temporal mean period
            critval_stn = runstn.copy() #get a new xarray data array from an existing one
            critval_stn[:] = 2 / np.sqrt(len(model)-1) #fill this new xarray data array with the critival value for a significant signal-to-noise ratio, as defined by the first equation in https://doi.org/10.1007/s00382-010-0977-x (Deser et al. 2012, Climate Dynamics)
            
            #caclulated raw / non transformed running LWT frequencies
            means = wt_agg.mean(axis=0)
            runmeans = means.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values of the yearly ensemble mean values
            runmeans_i = wt_agg.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values for each member
            
            # mean_i =  np.tile(runmeans_i.mean(dim='time'),(len(years),1)).transpose() # calculate the member-wise temporal mean values (anomalies w.r.t member-wise temporal mean)
            # #calculate anomalies w.r.t to global ensemble mean or member-wise temporal mean
            # if center_wrt == 'ensemble_mean':
                # runmeans_i_anom = runmeans_i - runmeans.mean() # calculate the running mean values for each individual model run minus the overall ensemble and temporal mean values (anomalies w.r.t. overall ensemble mean)
            # elif center_wrt == 'memberwise_mean':
                # runmeans_i_anom = runmeans_i - mean_i # calculate the running mean values for each individual model run minus the member-wise temporal mean values (anomalies w.r.t member-wise temporal mean)
            # else:
                # raise Exception('ERROR: check entry for <center_wrt> input parameter !')

            # runsignal = runmeans_i_anom.mean(dim='member').rename('signal') #ensemble signal for each temporal mean period
            # runnoise = runmeans_i_anom.std(dim='member').rename('noise') #ensemble standard deviation for each temporal mean period
            # runstn = np.abs(runsignal / runnoise).rename('signal-to-noise') #signal-to-noise ration for each temporal mean period
            # critval_stn = runstn.copy() #get a new xarray data array from an existing one
            # critval_stn[:] = 2 / np.sqrt(len(model)-1) #fill this new xarray data array with the critival value for a significant signal-to-noise ratio, as defined by the first equation in https://doi.org/10.1007/s00382-010-0977-x (Deser et al. 2012, Climate Dynamics)
            
            #plot signal-to-noise time series individually for each experiment and city, and compare with critical value
            min_occ = runstn.copy() #get a new xarray data array from an existing one
            min_occ[:] = wt_agg.min().values #overall minimum yearly WT frequency of all runs, used for plotting purpose below            
            fig = plt.figure()
            critval_stn.plot()
            runstn.plot()
            #runnoise.plot()
            if experiment[en] == 'dcppA':
                savename_stn = timeseries_dir+'/STN_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
            else:
                savename_stn = timeseries_dir+'/STN_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
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
                savename_ts = timeseries_dir+'/timeseries_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
            else:
                savename_ts = timeseries_dir+'/timeseries_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(start_year_step)+'_'+str(end_year_step)+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
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
            stn_all[lt,en,cc,:] = nanseries_stn
            signal_all[lt,en,cc,:] = nanseries_signal
            noise_all[lt,en,cc,:] = nanseries_noise
            runmeans_all[lt,en,cc,:] = nanseries_runmeans
            runmeans_i_all[lt,en,:,cc,:] = nanseries_runmeans_i
            wt_agg_tmean_all[lt,en,:,cc] = wt_agg_tmean
            #assign mrun, will be overwritten in each loop through lead_time

#convert output numpy arrays to xarray data array
stn_all = xr.DataArray(stn_all,coords=[lead_time,experiment,city,study_years],dims=['lead_time','experiment','city','time'],name='signal-to-noise')
signal_all = xr.DataArray(signal_all,coords=[lead_time,experiment,city,study_years],dims=['lead_time','experiment','city','time'],name='signal')
noise_all = xr.DataArray(noise_all,coords=[lead_time,experiment,city,study_years],dims=['lead_time','experiment','city','time'],name='noise')
runmeans_all = xr.DataArray(runmeans_all,coords=[lead_time,experiment,city,study_years],dims=['lead_time','experiment','city','time'],name='running_ensemble_mean')
runmeans_i_all = xr.DataArray(runmeans_i_all,coords=[lead_time,experiment,np.arange(len(mrun)),city,study_years],dims=['lead_time','experiment','member','city','time'],name='running_member_mean')
wt_agg_tmean_all = xr.DataArray(wt_agg_tmean_all,coords=[lead_time,experiment,np.arange(len(mrun)),city],dims=['lead_time','experiment','member','city'],name='member_mean')
stn_all.experiment.attrs['ensemble'] = ensemble
signal_all.experiment.attrs['ensemble'] = ensemble
noise_all.experiment.attrs['ensemble'] = ensemble
runmeans_all.experiment.attrs['ensemble'] = ensemble
runmeans_i_all.experiment.attrs['ensemble'] = ensemble
wt_agg_tmean_all.experiment.attrs['ensemble'] = ensemble

#for signal, noise and signal-to-noise ratios, the reanalysis is excluded, the focus is put on model experiments
stn_model = stn_all.sel(experiment=['dcppA','historical'])
noise_model = noise_all.sel(experiment=['dcppA','historical'])
wt_agg_tmean_model = wt_agg_tmean_all.sel(experiment=['dcppA','historical'])
stn_model_max = stn_model.max().values #maximum signal-to-noise ratio across all model experiments and lead-times
noise_model_min = noise_model.min().values
noise_model_max = noise_model.max().values #currently not in use yet, maximum noise / standard deviation across all model experiments and lead-times

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
        for exp in np.arange(len(experiment)):
            stn_all.sel(lead_time=lead_time[lt],experiment=experiment[exp],city=city[cc]).plot()
        plt.ylim(0,stn_all.max().values)
        plt.title(wtlabel.replace(" ","_")+'_FY'+str(lead_time[lt])+', '+city[cc]+', '+str(ensemble[0])+', '+str(len(mrun))+' members each')
        plt.legend(['critval']+experiment)
        savename_stn_city = figs+'/'+model[mm]+'/timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_stn_city,dpi=dpival)
        plt.close('all')
        del(fig)
        
        #plot the yearly and running temporal mean time series for each individual member and for the ensemble mean
        fig = plt.figure()
        study_years_mat = np.transpose(np.tile(study_years,[len(model),1]))
        for en in np.arange(len(ensemble)):
            plt.plot(study_years,runmeans_all[lt,en,cc,:].values,linewidth=2,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
            plt.plot(study_years_mat,np.transpose(runmeans_i_all[lt,en,:,cc,:].values),linewidth=0.5,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
            #plt.plot(years_mat,np.transpose(runmeans_i.values),linewidth=1)
        
        #set x and y axis configuration
        if axes_config == 'equal':
            #the same range of years is plotted for each lead-time (or forecast year), considering the longest period without NaNs
            plt.xticks(ticks=study_years[t_startindex:t_endindex][9::10],labels=study_years[t_startindex:t_endindex][9::10])
            plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
            
            # #y limits correspond to the minimum and maximum value of the running-mean relative frequency of the requested LWT in the specific city
            # plt.ylim([runmeans_i_all.sel(city=city[cc]).min().values,runmeans_i_all.sel(city=city[cc]).max().values])
            
            #equal y axis limit for all experiments of the same city
            plt.ylim([runmeans_i_all.sel(city=city[cc]).min().values,runmeans_i_all.sel(city=city[cc]).max().values])
        elif axes_config == 'individual':
            print('The x-axis in the time-series plot is optimized for lead-time = '+str(lead_time[lt])+' years.')
        else:
            raise Exception('ERROR: unknown entry for <axes_config> input parameter !')
        
        #plot corrleation coefficients in the titles
        rho1 = xr.corr(runmeans_all.sel(lead_time=lead_time[lt],experiment='dcppA',city=city[cc]),runmeans_all.sel(lead_time=lead_time[lt],experiment='historical',city=city[cc])) #calculate the correlation between the two running ensemble decadal mean time series
        rho2 = xr.corr(runmeans_all.sel(lead_time=lead_time[lt],experiment='dcppA',city=city[cc]),runmeans_all.sel(lead_time=lead_time[lt],experiment='20c',city=city[cc]))
        rho3 = xr.corr(runmeans_all.sel(lead_time=lead_time[lt],experiment='historical',city=city[cc]),runmeans_all.sel(lead_time=lead_time[lt],experiment='20c',city=city[cc]))
        rho1_all[lt,cc] = rho1
        rho2_all[lt,cc] = rho2
        rho3_all[lt,cc] = rho3
        plt.title(wtlabel.replace(" ","_")+', FY'+str(lead_time[lt])+', '+city[cc]+': r(dcppA-hist) = '+str(rho1.round(2).values)+', r(dcppA-obs) = '+str(rho2.round(2).values)+', r(hist-obs) = '+str(rho3.round(2).values), size = 8)
        # text_x = np.percentile(study_years,50) # x coordinate of text inlet
        # text_y = runmeans_i_all.max() - (runmeans_i_all.max() - runmeans_i_all.min())/25 # y coordinate of text inlet
        # plt.text(text_x,text_y, 'rho = '+str(np.round(rho,3)),size=8)
        
        #save the figure
        savename_ts_all = figs+'/'+model[mm]+'/timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
        plt.savefig(savename_ts_all,dpi=dpival)
        plt.close('all')
        del(fig)

    #plot summary city-scale results, difference in signal-to-noise ratio dcppA minus historical in pcolor format
    fig = plt.figure()
    stn_diff = stn_all.sel(lead_time=lead_time[lt], experiment='dcppA') - stn_all.sel(lead_time=lead_time[lt], experiment='historical')
    stn_diff.plot() #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
    #xlim2pcolor = [study_years[~np.isnan(stn_diff[0,:])][0]-0.5,study_years[~np.isnan(stn_diff[0,:])][-1]+0.5]
    #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
    plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
    plt.ylim(-0.5,len(city)-0.5)
    plt.title(wtlabel.replace(" ","_")+', SNR dcppa FY'+str(lead_time[lt])+' minus SNR historical')
    savename_diff_stn = figs+'/'+model[mm]+'/pcolor_diffstn_'+exp_label+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time[lt])+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
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
        stn_model.sel(lead_time=lead_time[lt], experiment=exp).plot(vmin=0,vmax=stn_model_max,edgecolors='k') #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
        
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
            savename_dcppa_stn = figs+'/'+model[mm]+'/pcolor_stn_'+exp+'_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            plt.title(wtlabel.replace(" ","_")+', '+exp+', FY'+str(lead_time[lt])+', SNR')
        else:
            #savename_dcppa_stn = figs+'/'+model[mm]+'/pcolor_stn_'+exp+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            savename_dcppa_stn = figs+'/'+model[mm]+'/pcolor_stn_'+exp+'_paired_with_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[-1])+'.'+outformat
            plt.title(wtlabel.replace(" ","_")+', '+exp+' paired with dcppA FY'+str(lead_time[lt])+', SNR')
        plt.savefig(savename_dcppa_stn,dpi=dpival)
        plt.close('all')
        del(fig)

    #Pcolor, difference between the standard deviation from dcppA with specific forecast year and historical for each running mean period
    fig = plt.figure()
    noise_diff = noise_all.sel(lead_time=lead_time[lt], experiment='dcppA') - noise_all.sel(lead_time = lead_time[lt], experiment='historical')
    noise_diff.plot() #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
    #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
    plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
    plt.ylim(-0.5,len(city)-0.5)
    plt.title(wtlabel.replace(" ","_")+', dcppa FY'+str(lead_time[lt])+' minus historical')
    savename_diff_noise = figs+'/'+model[mm]+'/pcolor_diffstd_dcppa_FY'+str(lead_time[lt])+'y_minus_hist_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_diff_noise,dpi=dpival)
    plt.close('all')
    del(fig)
    
    #Pcolor, standard deviation of dcppA with specific forecast year for each running mean period
    fig = plt.figure()
    noise_all.sel(lead_time=lead_time[lt], experiment='dcppA').plot(vmin=noise_model_min,vmax=noise_model_max)
    #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
    plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
    plt.ylim(-0.5,len(city)-0.5)
    plt.title(wtlabel.replace(" ","_")+', dcppa FY'+str(lead_time[lt]) + ' standard deviation')
    savename_dcppa_noise = figs+'/'+model[mm]+'/pcolor_std_dcppa_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_dcppa_noise,dpi=dpival)
    plt.close('all')
    del(fig)
    
    #Pcolor, standard deviation of historical experiment for each running mean period
    fig = plt.figure()
    noise_all.sel(lead_time=lead_time[lt], experiment='historical').plot(vmin=noise_model_min,vmax=noise_model_max)
    #plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
    plt.xlim([study_years[t_startindex:t_endindex].min()-0.5,study_years[t_startindex:t_endindex].max()+0.5])
    plt.ylim(-0.5,len(city)-0.5)
    plt.title(wtlabel.replace(" ","_")+', historical paired with dcppa FY'+str(lead_time[lt]) + ' standard deviation')
    #plt.gca().set_aspect('equal')
    savename_dcppa_noise = figs+'/'+model[mm]+'/pcolor_std_historical_paired_with_FY'+str(lead_time[lt])+'y_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
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
    
    #dcppA
    fig = plt.figure()
    wt_agg_tmean_all.sel(experiment='dcppA', city=city[cc]).plot(vmin=minval_tmean,vmax=maxval_tmean)
    plt.yticks(ticks = wt_agg_tmean_model.lead_time)
    plt.xticks(wt_agg_tmean_model.member.values, labels = mrun_dcppA,size=6,rotation=45)
    plt.xlabel('')
    plt.title(wtlabel.replace(" ","_")+', dcppA FY'+str(lead_time[lt]) + ' temporal mean')
    savename_dcppa_tmean = figs+'/'+model[mm]+'/pcolor_tmean_dcppa_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_dcppa_tmean,dpi=dpival)
    plt.close('all')
    del(fig)
    
    #historical
    fig = plt.figure()
    wt_agg_tmean_all.sel(experiment='historical', city=city[cc]).plot(vmin=minval_tmean,vmax=maxval_tmean)
    plt.yticks(ticks = wt_agg_tmean_model.lead_time)
    plt.xticks(wt_agg_tmean_model.member.values, labels = mrun_historical,size=6,rotation=45)
    plt.xlabel('')
    plt.title(wtlabel.replace(" ","_")+', dcppA FY'+str(lead_time[lt]) + ' temporal mean')
    savename_dcppa_tmean = figs+'/'+model[mm]+'/pcolor_tmean_historical_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_dcppa_tmean,dpi=dpival)
    plt.close('all')
    del(fig)
    
    #concat
    fig = plt.figure()
    wt_agg_tmean_concat.sel(city=city[cc]).plot(vmin=minval_tmean,vmax=maxval_tmean)
    std_step = wt_agg_tmean_std.sel(city=city[cc])
    yticks_tmean = wt_agg_tmean_concat.lead_time[0:len(lead_time)+1]
    ylabels_fy = list(wt_agg_tmean_all.sel(experiment='dcppA').lead_time.values.astype(str))+['hist']
    #ylabels_tmean = list(wt_agg_tmean_all.sel(experiment='dcppA').lead_time.values.astype(str))+['hist']
    ylabels_tmean = [ylabels_fy[ii]+' '+str(std_step[ii].round(2).values)[1:] for ii in np.arange(len(ylabels_fy))] #implicit loop to construct ylabels plus standard deviation of the temporal mean values from the individual members
    plt.yticks(ticks = yticks_tmean, labels = ylabels_tmean)
    plt.ylim(yticks_tmean[0]-0.5,yticks_tmean[-1]+0.5)
    plt.ylabel('forecast year and std of the mean')
    plt.title(wtlabel.replace(" ","_")+' in '+city[cc]+', dcppA plus hist, member-wise clim. mean freq.')
    savename_concat_tmean = figs+'/'+model[mm]+'/pcolor_tmean_concat_dcppa_historical_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_concat_tmean,dpi=dpival)
    plt.close('all')
    del(fig)

print('INFO: signal2noise_local.py has run successfully!')
