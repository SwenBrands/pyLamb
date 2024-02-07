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
taryears = [[1971,2010],[1971,2028],[1979,2014]] #list containing the start and end years for each ensemble, [1850,2261] for PiControl, [1901,2010] for 20c and historical, [1979,2014] or [1979,2017] for amip, [1971, 2028] for DCPPA

city = ['Barcelona','Bergen','Paris','Prague'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
#city = ['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
#city = ['Barcelona','Bergen'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest

tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #target months
lead_time = 10 #currently only used for experiment = dcppA; this is the lead time of the forecasts that were concatenated to form a single continuous time series in interpolator_xesmf.py
tarwts = [1] #[5,13,22] direcciones sur, [9,17,26] direcciones norte
center_wrt = 'ensemble_mean' # ensemble_mean or memberwise_mean; centering w.r.t. to ensemble (or overall) mean value or member-wise temporal mean value prior to calculating signal-to-noise

figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
meanperiod = 10 #running-mean period in years

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
yearly_units = '%' # count, % (relative frequency) or z-score; unit of the yearly LWT counts

#visualization options
plot_sig_stn_only = 'yes' #plot only significant signal-to-noise ratios in pcolor format, yes or no
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures

#auxiliary variables to be used in future versions of the script; currently not used
aggreg = 'year' #unlike map_lowfreq_var.py, this script currently only works for aggreg = 'year', other aggregation levels will be implemented in the future
anom = 'no' #not relevant here since yearly counts are considered, i.e. the annual cycle is not present in the time series, is kept for future versions of the script

#execute ###############################################################################################
if aggreg != 'year': #check for correct usage of the script
    raise Exception("Bad call of the script: This script currently only works for yearly LWT counts, i.e. aggreg = 'year' !)")

seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','')
if seaslabel == '123456789101112':
    seaslabel = 'yearly'

wtnames = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

study_years = np.arange(np.array(taryears).min(),np.array(taryears).max()+1,1) #numpy array of years limited by the lower and upper end of considered years 

#init output arrays of the ensemble x city loop series
#for 3d arrays ensemble x city x study_years
stn_all = np.zeros((len(ensemble),len(city),len(study_years)))
signal_all = np.copy(stn_all)
noise_all = np.copy(stn_all)
runmeans_all = np.copy(stn_all)

#loop throught the ensembles
for en in np.arange(len(ensemble)):
    #get ensemble configuration as defined in analysis_functions,py, see get_ensemble_config() function therein
    model,mrun,model_label,tarhours = get_ensemble_config(ensemble[en],experiment[en])
    #init 4d arrays ensemble x member x city x study_years
    if en == 0:
        runmeans_i_all = np.zeros((len(ensemble),len(mrun),len(city),len(study_years)))
    print('INFO: get annual and decadal mean time series as well as signal-to-noise ratios for '+aggreg+' LWT counts, '+ensemble[en]+' with '+str(len(mrun[en]))+' runs, '+str(tarmonths)+' months, '+str(taryears[en])+' years, '+str(tarhours)+' hours. Output LWT frequency units are: '+yearly_units+'.')
    wt_agg = np.zeros((len(model),len(list(range(taryears[en][0],taryears[en][1]+1)))))
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
            if experiment[en] == '20c':
                print('get_historical_metadata.py is not called for '+ensemble[en])
                file_taryears, timestep = get_target_period(model[mm],experiment[en])
            elif experiment[en] in ('amip','dcppA','historical','piControl'):
                runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm]) #check whether historical GCM configurations agree with those used in DCPPA ! 
                file_taryears, timestep = get_target_period(model[mm],experiment[en],cmip)
            else:
                raise Exception('ERROR: unknown entry in <experiment> input parameter')
            file_startyear = file_taryears[0]
            file_endyear = file_taryears[1]
            
            store_wt = store_wt_orig+'/'+timestep+'/'+experiment[en]+'/'+hemis
            if experiment[en] == 'dcppA':
                wt_file = store_wt+'/wtseries_'+model[mm]+'_'+experiment[en]+'_'+mrun[mm]+'_'+hemis+'_'+str(lead_time)+'y_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the LWT catalogues
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
                year_ind_wt = np.where((dates_wt.year >= taryears[en][0]) & (dates_wt.year <= taryears[en][1]) & (np.isin(dates_wt.hour,tarhours_alt)))[0]
            else:
                year_ind_wt = np.where((dates_wt.year >= taryears[en][0]) & (dates_wt.year <= taryears[en][1]) & (np.isin(dates_wt.hour,tarhours)))[0]

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
        means = wt_agg.mean(axis=0)
        runmeans = means.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values of the yearly ensemble mean values
        runmeans_i = wt_agg.rolling(time=meanperiod,center=True,min_periods=None).mean() # calculate running temporal mean values for each member
        runmeans_per_run =  np.tile(runmeans_i.mean(dim='time'),(len(years),1)).transpose() # calculate the member-wise temporal mean values (anomalies w.r.t member-wise temporal mean)
        #calculate anomalies w.r.t to global ensemble mean or member-wise temporal mean
        if center_wrt == 'ensemble_mean':
            runmeans_i_anom = runmeans_i - runmeans.mean() # calculate the running mean values for each individual model run minus the overall ensemble and temporal mean values (anomalies w.r.t. overall ensemble mean)
        elif center_wrt == 'memberwise_mean':
            runmeans_i_anom = runmeans_i - runmeans_per_run # calculate the running mean values for each individual model run minus the member-wise temporal mean values (anomalies w.r.t member-wise temporal mean)
        else:
            raise Exception('ERROR: check entry for <center_wrt> input parameter !')

        runsignal = runmeans_i_anom.mean(dim='member').rename('signal') #ensemble signal for each temporal mean period
        runnoise = runmeans_i_anom.std(dim='member').rename('noise') #ensemble standard deviation for each temporal mean period
        runstn = np.abs(runsignal / runnoise).rename('signal-to-noise') #signal-to-noise ration for each temporal mean period
        critval_stn = runstn.copy() #get a new xarray data array from an existing one
        critval_stn[:] = 2 / np.sqrt(len(model)-1) #fill this new xarray data array with the critival value for a significant signal-to-noise ratio, as defined by the first equation in https://doi.org/10.1007/s00382-010-0977-x (Deser et al. 2012, Climate Dynamics)
        min_occ = runstn.copy() #get a new xarray data array from an existing one
        min_occ[:] = wt_agg.min().values #overall minimum yearly WT frequency of all runs, used for plotting purpose below
        fig = plt.figure()
        critval_stn.plot()
        runstn.plot()
        #runnoise.plot()
        if experiment[en] == 'dcppA':
            savename_stn = timeseries_dir+'/STN_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(taryears[en][0])+'_'+str(taryears[en][1])+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
        else:
            savename_stn = timeseries_dir+'/STN_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel+'_ctr_'+center_wrt+'_'+str(taryears[en][0])+'_'+str(taryears[en][1])+'_'+seaslabel+'_'+aggreg+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
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
        plt.xlabel('year')
        plt.ylabel(ylabel_ts)    
        plt.xticks(ticks=years[9::10],labels=years[9::10])
        plt.xlim([years.min(),years.max()])
        plt.ylim([wt_agg.min(),wt_agg.max()])
        plt.title('LWT '+wtlabel+' '+city[cc]+' '+model_label+' '+str(len(mrun))+' members '+str(taryears[en][0])+'-'+str(taryears[en][1]))
        text_x = np.percentile(years,30) # x coordinate of text inlet
        text_y = wt_agg.values.max() - (wt_agg.values.max() - wt_agg.values.min())/25 # y coordinate of text inlet
        #plt.text(text_x,text_y, '$\sigma$ / $\mu$ = '+str(np.round(np.nanstd(runmeans)/np.nanmean(runmeans),3)),size=8) #plot standard deviation of running ensemble mean time series as indicator of forced response
        plt.text(text_x,text_y, 'temporal mean and minimum $\sigma$ = '+str(np.round(np.nanmean(runnoise),3))+' and '+str(np.round(np.nanmin(runnoise),3)),size=8) #plot temporal mean of the running standard deviation of the decadal mean LWT frequency anomalies from each member
        if experiment[en] == 'dcppA':
            savename_ts = timeseries_dir+'/timeseries_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(taryears[en][0])+'_'+str(taryears[en][1])+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
        else:
            savename_ts = timeseries_dir+'/timeseries_'+model[mm]+'_'+experiment[en]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(taryears[en][0])+'_'+str(taryears[en][1])+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
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
        stn_all[en,cc,:] = nanseries_stn
        signal_all[en,cc,:] = nanseries_signal
        noise_all[en,cc,:] = nanseries_noise
        runmeans_all[en,cc,:] = nanseries_runmeans
        runmeans_i_all[en,:,cc,:] = nanseries_runmeans_i

#convert output numpy arrays to xarray data array
stn_all = xr.DataArray(stn_all,coords=[experiment,city,study_years],dims=['experiment','city','time'],name='signal-to-noise')
signal_all = xr.DataArray(signal_all,coords=[experiment,city,study_years],dims=['experiment','city','time'],name='signal')
noise_all = xr.DataArray(noise_all,coords=[experiment,city,study_years],dims=['experiment','city','time'],name='noise')
runmeans_all = xr.DataArray(runmeans_all,coords=[experiment,city,study_years],dims=['experiment','city','time'],name='running_ensemble_mean')
runmeans_i_all = xr.DataArray(runmeans_i_all,coords=[experiment,np.arange(len(mrun)),city,study_years],dims=['experiment','member','city','time'],name='running_member_mean')
stn_all.experiment.attrs['ensemble'] = ensemble
signal_all.experiment.attrs['ensemble'] = ensemble
noise_all.experiment.attrs['ensemble'] = ensemble
runmeans_all.experiment.attrs['ensemble'] = ensemble
runmeans_i_all.experiment.attrs['ensemble'] = ensemble

#plot city-scale results
critvals_study_period = np.tile(critval_stn[0].values,len(study_years))
exp_label = str(experiment).replace('[','').replace(']','').replace("'","").replace(', ','_')
#detailed results for each city
rho1_all = np.zeros(len(city))
rho2_all = np.zeros(len(city))
rho3_all = np.zeros(len(city))
for cc in np.arange(len(city)):
    fig = plt.figure()
    plt.plot(stn_all.time.values,critvals_study_period)
    for exp in np.arange(len(experiment)):
        stn_all.sel(city=city[cc],experiment=experiment[exp]).plot()
    plt.ylim(0,stn_all.max().values)
    plt.title(city[cc]+', '+str(ensemble[0])+', '+str(len(mrun))+' members each')
    plt.legend(['critval']+experiment)
    savename_stn_city = figs+'/'+model[mm]+'/timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_stn_city,dpi=dpival)
    plt.close('all')
    del(fig)
    
    #plot the yearly and running temporall mean time series for each individual member and for the ensemble mean
    fig = plt.figure()
    study_years_mat = np.transpose(np.tile(study_years,[len(model),1]))
    for en in np.arange(len(ensemble)):
        plt.plot(study_years,runmeans_all[en,cc,:].values,linewidth=2,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
        plt.plot(study_years_mat,np.transpose(runmeans_i_all[en,:,cc,:].values),linewidth=0.5,color=ensemble_color[en],linestyle=ensemble_linestyle[en])
        #plt.plot(years_mat,np.transpose(runmeans_i.values),linewidth=1)
    
    #plot corrleation coefficients in the titles
    rho1 = xr.corr(runmeans_all.sel(experiment='dcppA',city=city[cc]),runmeans_all.sel(experiment='historical',city=city[cc])) #calculate the correlation between the two running ensemble decadal mean time series
    rho2 = xr.corr(runmeans_all.sel(experiment='dcppA',city=city[cc]),runmeans_all.sel(experiment='20c',city=city[cc]))
    rho3 = xr.corr(runmeans_all.sel(experiment='historical',city=city[cc]),runmeans_all.sel(experiment='20c',city=city[cc]))
    rho1_all[cc] = rho1
    rho2_all[cc] = rho2
    rho3_all[cc] = rho3
    plt.title(city[cc]+': r(dcppA-hist) = '+str(rho1.round(2).values)+', r(dcppA-obs) = '+str(rho2.round(2).values)+', r(hist-obs) = '+str(rho3.round(2).values), size = 8)
    # text_x = np.percentile(study_years,50) # x coordinate of text inlet
    # text_y = runmeans_i_all.max() - (runmeans_i_all.max() - runmeans_i_all.min())/25 # y coordinate of text inlet
    # plt.text(text_x,text_y, 'rho = '+str(np.round(rho,3)),size=8)
    
    #save the figure
    savename_ts_all = figs+'/'+model[mm]+'/timeseries_'+exp_label+'_'+city[cc]+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
    plt.savefig(savename_ts_all,dpi=dpival)
    plt.close('all')
    del(fig)

#plot summary city-scale results (difference between dcppA and historical) in pcolor format, difference in signal-to-noise ratio
fig = plt.figure()
stn_diff = stn_all.sel(experiment='dcppA')-stn_all.sel(experiment='historical')
xlim2pcolor = [study_years[~np.isnan(stn_diff[0,:])][0]-0.5,study_years[~np.isnan(stn_diff[0,:])][-1]+0.5]
stn_diff.plot() #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
plt.ylim(-0.5,len(city)-0.5)
plt.title('dcppa minus historical')
savename_diff_stn = figs+'/'+model[mm]+'/pcolor_diffstn_'+exp_label+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
plt.savefig(savename_diff_stn,dpi=dpival)
plt.close('all')
del(fig)

#plot signal-to-noise ratio separately for dcppA and historical
stn_all_model = stn_all.sel(experiment=['dcppA','historical']) #exclude the reanalysis, focus on model experiments
if plot_sig_stn_only == 'yes':
    print('WARNING: Only significant signal-to-noise ratios are plotted in pcolor format!')
    stn_all_model = stn_all_model.where(stn_all_model > critval_stn[0])

stn_model_max = stn_all_model.max().values #maximum signal-to-noise ratio across all model experiments
for exp in stn_all_model.experiment.values:
    fig = plt.figure()
    xlim2pcolor = [study_years[~np.isnan(stn_diff[0,:])][0]-0.5,study_years[~np.isnan(stn_diff[0,:])][-1]+0.5]    
    stn_all_model.sel(experiment=exp).plot(vmin=0,vmax=stn_model_max,edgecolors='k') #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
    plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
    plt.ylim(-0.5,len(city)-0.5)
    #set title and savename
    if exp == 'dcppA':
        savename_dcppa_stn = figs+'/'+model[mm]+'/pcolor_stn_'+exp+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
        plt.title(exp+', forecast year '+str(lead_time))
    else:
        savename_dcppa_stn = figs+'/'+model[mm]+'/pcolor_stn_'+exp+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
        plt.title(exp)
    plt.savefig(savename_dcppa_stn,dpi=dpival)
    plt.close('all')
    del(fig)

#summary results (difference between dcppA and historical) in pcolor format, difference in noise / standard deviation
fig = plt.figure()
noise_diff = noise_all.sel(experiment='dcppA')-noise_all.sel(experiment='historical')
noise_diff.plot() #plots a pcolor of the differences in the signal-to-noise ratios along the time axis (10-yr running differences)
plt.xlim(xlim2pcolor[0],xlim2pcolor[1])
plt.ylim(-0.5,len(city)-0.5)
plt.title('dcppa minus historical')
savename_diff_noise = figs+'/'+model[mm]+'/pcolor_diffnoise_'+exp_label+'_'+model[mm]+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(lead_time)+'y_ctr_'+center_wrt+'_'+str(study_years[0])+'_'+str(study_years[1])+'.'+outformat
plt.savefig(savename_diff_noise,dpi=dpival)
plt.close('all')
del(fig)

print('INFO: signal2noise_local.py has run successfully!')
