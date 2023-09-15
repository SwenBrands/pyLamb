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

#set input parameter
city = ['Athens','Azores','Casablanca','Cairo','Prague','SantiagoDC','Barcelona'] #city or point of interest
tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #target months
taryears = [1901,2010] #start and end year
tarwts = [18] #[5,13,22] direcciones sur, [9,17,26] direcciones norte
dpival = 300
outformat = 'png'
figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
wtnames = ['PA','DANE','DAE','DASE','DAS','DASW','DAW','DANW','DAN','PDNE','PDE','PDSE','PDS','PDSW','PDW','PDNW','PDN','PC','DCNE','DCE','DCSE','DCS','DCSW','DCW','DCNW','DCN','U']

#model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
#mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']

model = ['mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr']
mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']

experiment = 'historical' #historical, 20c, amip, ssp245, ssp585
meanperiod = 10 #running-mean period in years

#options used for periodgram, experimental so far, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
periodogram = 'yes'
fs = 1 #sampling frequency for 3-hourly data, 30*8 is monthly, 90*8 is seasonal, alternatively 1 for yearly data
nfft_quotient = None #integer setting the quotient or None for default options, this is the fraction of the time series used as lenght of the FFT, 4 is recommended by Schönwiese 2006; if set to None, the default nfft option is used in signal.periodgram(). This option is not used in signal.csd() which always assumes None / default options 
window = 'hamming' #http://qingkaikong.blogspot.com/2017/01/signal-processing-finding-periodic.html
scaling = 'spectrum'
repetitions = 10000
detrend = 'linear' #False, linear or constant for removing the linear trend or mean only prior to calculating the power spectrum
ci_percentiles = [2.5,5.,95.,97.5]
cutoff = [] #defines where the x-axis is cut-off in the time-series plots, [] for no cutoff

#visualization options
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures

#auxiliary variables to use with plot_power_spectrum() function defined in <analysis_functions.py>
aggreg = 'year'
anom = 'no'

#execute ###############################################################################################
exec(open('analysis_functions.py').read())
seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','')
wt_names = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")

wt_center_yr = np.zeros((len(model),len(list(range(taryears[0],taryears[1]+1)))))
for cc in np.arange(len(city)):
    print('INFO: obtain results for '+city[cc]+'!')
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
        # if model[mm] == 'cera20c':
            # timestep = '3h'
        # else:
            # timestep = '6h'
        if model[mm] == 'cera20c':
            timestep = '3h'
            file_startyear = '1901'
            file_endyear = '2010'
        elif model[mm] == 'era5':
             timestep = '6h'
             file_startyear = '1979'
             file_endyear = '2020'
        elif model[mm] in ('mpi_esm_1_2_hr','ec_earth3_veg'):
             timestep = '6h'
             file_startyear = '1850'
             file_endyear = '2014'
        else:
            timestep = '6h'
            file_startyear = '1979'
            file_endyear = '2005'
            
        store_wt = store_wt_orig+'/'+timestep+'/'+experiment+'/'+hemis
        wt_file = store_wt+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(file_startyear)+'_'+str(file_endyear)+'.nc' #path to the LWT catalogues

        #load the LWT time series for the centerlat and target years obtained above
        wt = xr.open_dataset(wt_file)
        
        #get the gridbox nearest to tarlon and tarlat
        lon_center = wt.lon.sel(lon=tarlon,method='nearest').values
        lat_center = wt.lat.sel(lat=tarlat,method='nearest').values
        #lon_center = np.array(lon_center-180.)
        
        wt_center = wt.sel(lon=lon_center,lat=lat_center,method='nearest')
        dates_wt = pd.DatetimeIndex(wt_center.time.values)
        year_ind_wt = np.where((dates_wt.year >= taryears[0]) & (dates_wt.year <= taryears[1]))[0]
        wt_center = wt_center.isel(time=year_ind_wt)
        #wt_val = wt_center.wtseries.values

        bin_array = np.zeros(wt_center.wtseries.shape)
        tarwt_ind = np.where(wt_center.wtseries.isin(tarwts))
        bin_array[tarwt_ind] = 1
        arr_tarwts = xr.DataArray(data=bin_array,coords=[pd.DatetimeIndex(wt_center.time)],dims='time',name='wtseries')
        #get time series with yearly target WT counts
        wt_center_yr_step = arr_tarwts.groupby('time.year').sum('time')
        wt_center_yr[mm,:] = wt_center_yr_step.values

    #get running mean of yearly ensemble mean counts
    years = wt_center_yr_step.year.values
    means = np.mean(wt_center_yr,axis=0)
    means = xr.DataArray(means,coords=[years],dims=['time'],name='wtfreq')
    runmeans = means.rolling(time=meanperiod,center=True,min_periods=None).mean()
    
    #create target directories for periodogram if missing
    periodogram_dir = figs+'/'+model[mm]+'/local/'+aggreg+'/periodogram/'
    if os.path.isdir(periodogram_dir) != True:
        os.makedirs(periodogram_dir) 
    
    #experimental part of the script, get the frequency and spectrum for yearly ensemble mean WT counts
    if periodogram == 'yes':
        ## get the frequency and spectrum for yearly ensemble mean WT counts
        if nfft_quotient == None:
            print('Info: The default nfft options in signal.periodogram() and signal.csd() will be used for (cross) spectral analysis...')
            nfft = None
        elif nfft_quotient > 0:
            nfft = int(round(len(means)/nfft_quotient)) #use optionally in the next 4 lines
            print('Info: As requested by the user, '+str(nfft)+' ffts will be used for spectral analysis and the default option for cross-spectral analysis...')
        else:
            raise Exception('ERROR: unknown entry for <nfft_quotient>!')
        
        #get power spectrum
        f, Pxx = signal.periodogram(means, fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend) #transforms to anomalies prior to calculating frequencies and spectrum by default

        #get confidence intervals from randomly reshuffled time series
        ntime = len(means) #recall that means is an xr DataArray
        #init array to be filled with power spectra from randomly reshuffled time series
        Pxx_rand = np.zeros((repetitions,len(Pxx)))
        print('INFO: Init power spectrum calculations with randomly reshuffled times series and '+str(repetitions)+' repetitions...')
        for rr in list(np.arange(repetitions)):
            rand1 = np.random.randint(0,ntime,ntime)
            f_rand_step, Pxx_rand_step = signal.periodogram(means[rand1], fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend)
            Pxx_rand[rr,:] = Pxx_rand_step
        print('Info: Reshuffling has terminated!')
        #get 95% confidence intervals for the random power spectra
        ci = np.percentile(Pxx_rand,ci_percentiles,axis=0)
        ##define aux variables for plotting
        #savename = figs+'/'+model[mm]+'/local/'+aggreg+'/periodogram/'+model[mm]+'_'+str(len(mrun))+'_members_periodogram_LWT_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
        savename = periodogram_dir+'/'+model[mm]+'_'+str(len(mrun))+'_members_periodogram_LWT_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
        titlelabel = 'PS '+wtlabel+' '+model[mm].upper()+' '+str(len(mrun))+'m '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg+' nfft'+str(nfft)+' dtrend'+detrend+' anom'+anom+' '+city[cc]
        #plot_power_spectrum('single',f,Pxx,ci,fs,window,nfft,detrend,anom,wtnames,tarwts,model[0],'10',taryears,seaslabel,aggreg,city,outformat,dpival,titlesize)
        plot_power_spectrum('single',f,Pxx,ci,fs,window,nfft,detrend,aggreg,savename,titlelabel,dpival,titlesize)

    #plot year-to-year WT counts for each ensemble member of CERA-20C and the <meanperiod>-year rolling temporal average of the yearly ensemble mean counts
    timeseries_dir = figs+'/'+model[0]+'/local/'+aggreg+'/timeseries'
    #create target directory if missing
    if os.path.isdir(timeseries_dir) != True:
        os.makedirs(timeseries_dir)
        
    fig = plt.figure()
    plt.plot(np.transpose(np.tile(years,[10,1])),np.transpose(wt_center_yr),linewidth=1)
    plt.plot(years[len(years)-len(runmeans):],runmeans,linewidth=2,color='red')
    plt.xlabel('year')
    plt.ylabel('Yearly occurrence frequency based on '+str(timestep)+'h data')
    plt.xticks(ticks=years[9::10],labels=years[9::10])
    plt.xlim([years.min(),years.max()])
    plt.ylim([wt_center_yr.min()-10,wt_center_yr.max()+10])
    plt.title('LWT '+wtlabel+' '+model[mm].upper()+' '+str(len(mrun))+' members '+str(taryears[0])+'-'+str(taryears[1])+' '+city[cc])
    #savename = figs+'/'+model[0]+'/local/'+aggreg+'/timeseries/'+model[mm]+'_'+str(len(mrun))+'_members_intan_var_LWT_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
    savename = timeseries_dir+'/'+model[mm]+'_'+str(len(mrun))+'_members_intan_var_LWT_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
    plt.savefig(savename,dpi=dpival)
    plt.close('all')
    wt_center.close()
    wt.close()

print('INFO: get_decvar_local_ensmean.py has run successfully!')
