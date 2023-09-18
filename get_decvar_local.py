# -*- coding: utf-8 -*-

''''This script plots composites maps for the 27 LWTs for a given location and time period'''

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
ensemble = 'miroc6' #cera20c or mpi_esm_1_2_hr
experiment = 'amip' #historical, amip or 20c
city = ['Barcelona','Bergen','Paris','Prague'] #['Athens','Azores','Barcelona','Bergen','Cairo','Casablanca','Paris','Prague','SantiagoDC','Seattle','Tokio'] #city or point of interest
tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #target months
taryears = [1979,2014] #start and end year, [1901,2010] for 20c and historical, [1979,2014] for amip
tarhours = [0,6,12,18]
tarwts = [1] #[5,13,22] direcciones sur, [9,17,26] direcciones norte
nfft_quotients = [1] # n / nff_quotient equals the length of the maximum period; nfft_quotient = number of non-overlapping sub-periods used by the Welch method

figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'

meanperiod = 10 #running-mean period in years

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
periodogram_type = 'periodogram' #Welch or periodogram
fs = 1 #sampling frequency for 3-hourly data, 30*8 is monthly, 90*8 is seasonal, alternatively 1 for yearly data
window = 'hann' #hann, nuttall etc. http://qingkaikong.blogspot.com/2017/01/signal-processing-finding-periodic.html
scaling = 'spectrum'
repetitions = 10000 #10000 is ideal
detrend = 'linear' #linear or constant for removing the linear trend or mean only prior to calculating the power spectrum
ci_percentiles = [2.5,5.,10.,90.,95.,97.5] #these crtical values for power spectra will be calculated
ci_tar_percentile = 95. #and this one will be plotted
cutoff = [] #defines where the x-axis is cut-off in the time-series plots, [] for no cutoff
yearly_units = '%' # count, % (relative frequency) or z-score; unit of the yearly LWT counts
standardization = 'yes' #Are the count or % LWT time series z-transformed prior to calculating the power spectra, yes or no ?
reshuffling = 'ar1' #white or ar1
acorr_lag = 1 #lag of autocorrelation function to be used to generate autoregressive process used to obtain critical values for <reshuffling = 'ar1'>; not used for <reshuffling = 'white'>

#visualization options
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures

#auxiliary variables to be used in future versions of the script; currently not used
aggreg = 'year' #unlike map_lowfreq_var.py, this script currently only works for aggreg = 'year', other aggregation levels will be implemented in the future
anom = 'no' #not relevant here since yearly counts are considered, i.e. the annual cycle is not present in the time series, is kept for future versions of the script

#execute ###############################################################################################
if aggreg != 'year': #check for correct usage of the script
    raise Exception("Bad call of the script: This script currently only works for yearly LWT counts, i.e. aggreg = 'year' !)")
if (reshuffling == 'ar1') & (standardization != 'yes'):
    raise Exception("Bad call of the script: ar1 reshuffling currently only works for standardized (or z-transformed) LWT time series!") 

if ensemble == 'cera20c' and experiment == '20c':
    model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
    mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']
    model_label = 'CERA-20C'
elif ensemble == 'mpi_esm_1_2_hr' and experiment == 'historical':
    model = ['mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr']
    mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
    model_label = 'MPI-ESM1.2-HR'
elif ensemble == 'ec_earth3_veg' and experiment == 'historical':
    model = ['ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg']
    mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r10i1p1f1','r11i1p1f1']
    model_label = 'EC-Earth3-Veg'
elif ensemble == 'miroc6' and experiment == 'amip':
    model = ['miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6']
    mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
    model_label = 'MIROC6'
else:
    raise Exception('ERROR: unknown entry for <ensemble> and/or <experiment> input parameters !') 

print('INFO: get low frequency variability for '+aggreg+' LWT counts, '+ensemble+' with '+str(len(mrun))+' runs, '+str(tarmonths)+' months, '+str(taryears)+' years, '+str(tarhours)+' hours, ' +window+' window, '+detrend+' detrending and '+str(repetitions)+' repetitions for the resampling approach. Output units are: '+yearly_units+'.')

seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','')
if seaslabel == '123456789101112':
    seaslabel = 'yearly'

wtnames = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

wt_agg = np.zeros((len(model),len(list(range(taryears[0],taryears[1]+1)))))
for qq in np.arange(len(nfft_quotients)):
    nfft_quotient = nfft_quotients[qq]
    for cc in np.arange(len(city)):
        print('INFO: obtaining results for '+city[cc]+' and nfft_quotient '+str(nfft_quotient))
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
            #get metadata for this GCM
            runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm])
            #define the time period the GCM data is interpolated for as a function of the experiment and considered GCM
            file_taryears = get_target_period(model[mm],experiment,cmip)
            file_startyear = file_taryears[0]
            file_endyear = file_taryears[1]
            
            #get timestep
            if model[mm] == 'cera20c':
                timestep = '3h'
            elif model[mm] == 'era5':
                 timestep = '6h'
            elif model[mm] in ('mpi_esm_1_2_hr','ec_earth3_veg'):
                 timestep = '6h'
            else:
                timestep = '6h'
                
            store_wt = store_wt_orig+'/'+timestep+'/'+experiment+'/'+hemis
            wt_file = store_wt+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the LWT catalogues

            #load the LWT time series for the centerlat and target years obtained above
            wt = xr.open_dataset(wt_file)
            
            #get the gridbox nearest to tarlon and tarlat
            lon_center = wt.lon.sel(lon=tarlon,method='nearest').values
            lat_center = wt.lat.sel(lat=tarlat,method='nearest').values        
            wt_center = wt.sel(lon=lon_center,lat=lat_center,method='nearest')
            
            #select requested time period (years and hours)
            dates_wt = pd.DatetimeIndex(wt_center.time.values)
            year_ind_wt = np.where((dates_wt.year >= taryears[0]) & (dates_wt.year <= taryears[1]) & (np.isin(dates_wt.hour,tarhours)))[0]
            wt_center = wt_center.isel(time=year_ind_wt)

            #modify origional LWT time series containing 27 types to a binary absence (0) - occurrence (1) time series of the requested types only
            bin_array = np.zeros(wt_center.wtseries.shape)
            tarwt_ind = np.where(wt_center.wtseries.isin(tarwts))
            bin_array[tarwt_ind] = 1
            arr_tarwts = xr.DataArray(data=bin_array,coords=[pd.DatetimeIndex(wt_center.time)],dims='time',name='wtseries')
            
            #get time series with yearly target WT counts
            if aggreg == 'year':
                if yearly_units == 'count':
                    wt_agg_step = arr_tarwts.groupby('time.year').sum('time')
                elif yearly_units == '%':
                    hours_per_year = arr_tarwts.groupby('time.year').count()
                    wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                    wt_agg_step = wt_agg_step / hours_per_year *100
                elif yearly_units == 'z-score':
                    hours_per_year = arr_tarwts.groupby('time.year').count()
                    wt_agg_step = arr_tarwts.groupby('time.year').sum('time') #calculate annual mean values
                    wt_agg_step = z_transform(wt_agg_step / hours_per_year *100)
                else:
                    raise Exception('ERROR: unknown entry for <yearly_units>!')       
            ntime = wt_agg_step.values.shape[0]
            wt_agg[mm,:] = wt_agg_step.values #fill into the numpy array <wt_agg> for further processing
            
            #experimental part of the script, get the frequency and spectrum for yearly ensemble mean WT counts
            if nfft_quotient == None:
                print('Info: The default nfft options in signal.periodogram() will be used for spectral analysis...')
                nfft = None
            elif nfft_quotient > 0:
                nfft = int(np.floor(ntime/nfft_quotient)) #use optionally in the next 4 lines
                print('Info: As requested by the user, '+str(nfft)+' ffts will be used for spectral analysis...')
            else:
                raise Exception('ERROR: unknown entry for <nfft_quotient>!')
            
            #get power spectrum
            if (standardization == 'yes') and (yearly_units != 'z-score'):
                print('INFO: Power spectra will be calculated on '+yearly_units+' LWT times series that are z-transformed prior to calculating the power spectra...')
                wt_agg_to_period = z_transform(wt_agg[mm,:])
            elif (standardization == 'yes') and (yearly_units == 'z-score'): #in this case the z-transformation has been already performed above and is here not necessary any more.
                print('INFO: Power spectra will be calculated on '+yearly_units+' LWT times series that are z-transformed prior to calculating the power spectra...')
                wt_agg_to_period = wt_agg[mm,:]
            elif standarization == 'no':
                print('INFO: Power spectra will be calculated on non-transformed / raw LWT time series with unit '+yearly_units+'...')
                wt_agg_to_period = wt_agg[mm,:]
            else:
                raise Exception('ERROR: unknown entry for <standardization> !')
            if periodogram_type == 'periodogram':
                f_def, Pxx_def = signal.periodogram(wt_agg_to_period, fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend) #transforms to anomalies prior to calculating frequencies and spectrum by default
            elif periodogram_type == 'Welch':
                f_def, Pxx_def = signal.welch(wt_agg_to_period, fs=fs, window = window, nperseg=nfft, noverlap=None, nfft=None, detrend=detrend, return_onesided=True, scaling=scaling, average='mean')
            else:
                raise Exception('ERROR: unknown entry for <periodogram_type!')
                
            ##get AR1 process with lag-1 autocorrelation coefficient from wt_agg (containing the LWT time series relevant here), see https://goodboychan.github.io/python/datacamp/time_series_analysis/2020/06/08/01-Autoregressive-Models.html
            #acorr = sm.tsa.acf(wt_agg_to_period, nlags = ntime)
            acorr = sm.tsa.acf(signal.detrend(wt_agg_to_period,type=detrend), nlags = ntime)
            print('INFO: The lag '+str(acorr_lag)+' autocorr. coeff. of the '+wtlabel+' time series in '+city[cc]+', '+model[mm]+', '+mrun[mm]+', '+str(taryears)+', '+seaslabel+' is '+str(np.round(acorr[acorr_lag],4))) 
            ar1 = np.array([1, acorr[acorr_lag]*-1]) #ArmaProcess() functions requires to switch the sign of the autocorrelation coefficient 
            ma1 = np.array([1])
            AR_object1 = ArmaProcess(ar1, ma1)
            
            #initialize output arrays of spectral analysis
            if mm == 0:
                n_freq = len(f_def)
                fq =  np.zeros((len(model),n_freq))
                Pxx =  np.zeros((len(model),n_freq))
                Pxx_ci =  np.zeros((len(model),n_freq,len(ci_percentiles)))

            #get confidence intervals from randomly reshuffled time series
            #init array to be filled with power spectra from randomly reshuffled time series
            print('INFO: Init power spectrum calculations with randomly reshuffled times series following '+reshuffling+' noise and '+str(repetitions)+' repetitions...')
            Pxx_rand = np.zeros((repetitions,Pxx.shape[1]))
            if reshuffling == 'white': #get power spectrum from white noise data
                for rr in list(np.arange(repetitions)):
                    rand1 = np.random.randint(0,ntime,ntime)
                    if periodogram_type == 'periodogram':
                        f_rand_step, Pxx_rand_step = signal.periodogram(wt_agg_to_period[rand1], fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend)
                    elif periodogram_type == 'Welch':
                        f_rand_step, Pxx_rand_step = signal.welch(wt_agg_to_period[rand1], fs=fs, window = window, nperseg=nfft, noverlap=None, nfft=None, detrend=detrend, return_onesided=True, scaling=scaling, average='mean')
                    else:
                        raise Exception('ERROR: check entry for <periodogram_type>!')
                    Pxx_rand[rr,:] = Pxx_rand_step
            elif reshuffling == 'ar1': #get power spectrum from data obtained random AR1 data, must be z-transformed in any case
                for rr in list(np.arange(repetitions)):
                    rand_data = AR_object1.generate_sample(nsample=ntime) #generates red noise time series
                    if periodogram_type == 'periodogram':
                        f_rand_step, Pxx_rand_step = signal.periodogram(rand_data, fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend)
                    elif periodogram_type == 'Welch':
                        f_rand_step, Pxx_rand_step = signal.welch(rand_data, fs=fs, window = window, nperseg=nfft, noverlap=None, nfft=None, detrend=detrend, return_onesided=True, scaling=scaling, average='mean')
                    else:
                        raise Exception('ERROR: check entry for <periodogram_type>!')
                    Pxx_rand[rr,:] = Pxx_rand_step
                print('INFO: Reshuffling has terminated!')
            
            #get 95% confidence intervals for the random power spectra and fill the pre-initialized arrays
            Pxx_ci[mm,:,:] = np.transpose(np.percentile(Pxx_rand,ci_percentiles,axis=0))
            fq[mm,:] = f_def
            Pxx[mm,:] = Pxx_def

        #get running mean of yearly ensemble mean counts
        years = wt_agg_step.year.values
        wt_agg = xr.DataArray(wt_agg,coords=[np.arange(wt_agg.shape[0]),years],dims=['member','time'],name='wtfreq')
        means = wt_agg.mean(axis=0)
        #means = np.mean(wt_agg,axis=0)    
        #means = xr.DataArray(means,coords=[years],dims=['time'],name='wtfreq')
        runmeans = means.rolling(time=meanperiod,center=True,min_periods=None).mean()
        runmeans_i = wt_agg.rolling(time=meanperiod,center=True,min_periods=None).mean() # i stands for individual model run
            
        #create target directories for periodogram if missing
        periodogram_dir = figs+'/'+model[mm]+'/'+experiment+'/local/'+city[cc]+'/'+aggreg+'/periodogram/'
        if os.path.isdir(periodogram_dir) != True:
            os.makedirs(periodogram_dir)
        
        ##PLOT the results
        #set label for y coordinate of the time-series (suffix ts) and periodgram (suffix p) plot
        if yearly_units == 'count':
            ylabel_ts = 'Yearly occurrence frequency (count)'
        elif yearly_units == '%':
            ylabel_ts = 'Yearly relative occurrence frequency (%)'
        elif yearly_units == 'z-score':
            ylabel_ts = 'z-score'
        else:
            raise Exception('ERROR: unknown entry for <yearly_units> input parameter!')
        
        if standardization == 'yes':
            ylabel_p = 'Power spectrum amplitude (z-score$²$)'
        else:
            ylabel_p = 'Power spectrum amplitude ('+yearly_units+'$²$)'
        
        #periodogram
        period_yr = 1/fq[0,:]
        maxamp_ind = np.argsort(np.max(Pxx,axis=0))
        outind = period_yr[maxamp_ind] == np.inf #find inf values
        maxamp_ind = np.delete(maxamp_ind,outind) #and remove them
        fig = plt.figure()
        plt.plot(fq.transpose(), Pxx.transpose(),linewidth=0.5)
        #plot critcal values from randomly reshuffled data if available
        if len(Pxx_ci) > 0: #check whether Pxx_ci is empty ([])
            ci_ind = int(np.where(np.array(ci_percentiles) == ci_tar_percentile)[0])
            plt.plot(fq.transpose(), Pxx_ci[:,:,ci_ind].transpose(),color='red',linewidth=1,linestyle='dotted')
        
        #get number ensemble members that agree on a significant power spectra; do this for all periods / frequencies
        sigsum = np.sum(Pxx > Pxx_ci[:,:,ci_ind],axis=0)
        
        ##plot values as text for the 10 (or 5) largest amplitudes in the spectrum
        #for ii in maxamp_ind[-10:]: 
        for ii in maxamp_ind[-5:]:
            if aggreg == 'year':
                textme = str(np.round(period_yr[ii],2))+' ('+str(sigsum[ii])+')'
            else:
                textme = str(int(np.round(period_yr[ii])))+' ('+str(sigsum[ii])+')'
            plt.text(fq[0,ii],np.max(Pxx,axis=0)[ii],textme,size=8)
        
        #plot weighted ensemble spread score (espread_weighted) in upper right corner, excluding inf
        valind = np.where(period_yr != np.inf)
        #espread = (np.percentile(Pxx[:,valind],75,axis=0)-np.percentile(Pxx[:,valind],25,axis=0))/np.median(Pxx[:,valind],axis=0)
        espread = (np.percentile(Pxx[:,valind],75,axis=0)-np.percentile(Pxx[:,valind],25,axis=0))
        weights = np.sqrt(period_yr[valind])
        espread_weighted = np.sum(espread*weights)/(weights.sum())
        text_x = np.percentile(fq[0,:],77) # x coordinate of text inlet
        #text_y = Pxx.max() - (Pxx.max() - Pxx.min())/25 # y coordinate of text inlet
        text_y = Pxx.max() # y coordinate of text inlet
        plt.text(text_x, text_y, 'Spread Score = '+str(np.round(espread_weighted,3)),size=8) #plot Ensemble Convergence Score

        ##plot y-axis in log scale
        #plt.yscale('log')
        if aggreg == 'year':
            ag_label = 'years'
            x_labels = np.round(period_yr,2)
            rotation = 90.
        else:
            ag_label = aggreg+' days'
            x_labels = np.round(period_yr)
            rotation = 0.
            
        plt.xlabel('Period ('+ag_label+')')
        plt.xticks(ticks=fq[0,:],labels=x_labels,rotation=rotation,fontsize=6)
        plt.ylabel(ylabel_p)
        titlelabel = periodogram_type+' '+wtlabel+' '+city[cc]+' '+model_label+' '+str(len(mrun))+'m '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg+' '+window+' nfft'+str(nfft)+' dtrend'+detrend+' anom'+anom
        plt.title(titlelabel,size=titlesize)
        savename = periodogram_dir+'/'+periodogram_type+'_'+model[mm]+'_'+experiment+'_'+str(len(mrun))+'mem_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+reshuffling+'_'+seaslabel+'_'+aggreg+'_'+window+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'_lat'+str(wt_center.lat.values)+'.'+outformat
        plt.savefig(savename,dpi=dpival)
        plt.close('all')
        
        print('The 10 periods (in years) with largest absolute amplitude in ascending order are:')
        print(period_yr[maxamp_ind[-10:]])

        #Plot year-to-year WT counts for each ensemble member of CERA-20C and the <meanperiod>-year rolling temporal average of the yearly ensemble mean counts
        timeseries_dir = figs+'/'+model[mm]+'/'+experiment+'/'+model[0]+'/local/'+city[cc]+'/'+aggreg+'/timeseries'
        #create target directory if missing
        if os.path.isdir(timeseries_dir) != True:
            os.makedirs(timeseries_dir)
            
        fig = plt.figure()
        years_mat = np.transpose(np.tile(years,[10,1]))
        plt.plot(years_mat,np.transpose(wt_agg.values),linewidth=0.5,color='grey',linestyle='dotted')
        plt.plot(years,runmeans,linewidth=2,color='black')
        plt.plot(years_mat,np.transpose(runmeans_i.values),linewidth=1)
        plt.xlabel('year')
        plt.ylabel(ylabel_ts)    
        plt.xticks(ticks=years[9::10],labels=years[9::10])
        plt.xlim([years.min(),years.max()])
        plt.ylim([wt_agg.min(),wt_agg.max()])
        plt.title('LWT '+wtlabel+' '+city[cc]+' '+model_label+' '+str(len(mrun))+' members '+str(taryears[0])+'-'+str(taryears[1]))
        text_x = np.percentile(years,84) # x coordinate of text inlet
        text_y = wt_agg.values.max() - (wt_agg.values.max() - wt_agg.values.min())/25 # y coordinate of text inlet
        plt.text(text_x,text_y, '$\sigma$ / $\mu$ = '+str(np.round(np.nanstd(runmeans)/np.nanmean(runmeans),3)),size=8) #plot standard deviation of running ensemble mean time series as indicator of forced response
        savename = timeseries_dir+'/timeseries_'+model[mm]+'_'+experiment+'_'+str(len(mrun))+'mem_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
        plt.savefig(savename,dpi=dpival)
        plt.close('all')
        wt_center.close()
        wt.close()

print('INFO: get_decvar_local.py has run successfully!')
