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
import time
import pdb as pdb #then type <pdb.set_trace()> at a given line in the code below

#set input parameters
region1 = 'nh' #region 1 as defined in analysis_functions.py
region2 = 'sh_midlats' #region 2 as defined in analysis_functions.py
tarwts_various = [[7,15,24],[1],[18]] #list of lists, loop through various wt combinations for exploratory data analysis; e.g. [5,13,22] are southerly directions, [9,17,26] are northerly ones
# tarwts_various = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27]] #list of lists, loop through various wt combinations for exploratory data analysis; e.g. [5,13,22] are southerly directions, [9,17,26] are northerly ones
tarmonths_various = [[1,2,3,4,5,6,7,8,9,10,11,12]] #loop through various seasons for exploratory data analysis
# tarmonths_various = [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12]] #loop through various seasons for exploratory data analysis
taryears = [1950,2010] #start and end yeartaryears = [1979,2005] #start and end year
aggreg = 'year' #temporal aggregation of the 3 or 6-hourly time series: 'year' or '1', '7', '10' or '30' indicating days, must be string format
fig_root = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
wtnames = ['PA','DANE','DAE','DASE','DAS','DASW','DAW','DANW','DAN','PDNE','PDE','PDSE','PDS','PDSW','PDW','PDNW','PDN','PC','DCNE','DCE','DCSE','DCS','DCSW','DCW','DCNW','DCN','U']
ensemble_label = 'CERA-20C' #used for plotting purpose only
model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']
#model = ['cera20c','cera20c']
#mrun = ['m0','m1']
# model = ['era5']
# mrun = ['r1i1p1']
# experiment = 'historical' #historical, 20c, amip, ssp245, ssp585
experiment = '20c'
meanperiod = 10 #used to calculate the rolling mean WT counts for the temporally aggregated data as defined in <aggreg> above, .e.g. 10
latweighting = 'yes' #weight the point-wise WT counts by latitude prior to summing over the entire domain
anom = 'yes' #remove monthly mean values to obtain anomalies
rollw = 21 #rolling window in days used to calculate the climatological mean which is then removed from each entry of the daily time series; must be an uneven number

#options used for periodgram, experimental so far
fs = 1 #sampling frequency used for calcluating power and cross-power spectra; is applied on temporally aggregated values as defined by <aggreg> above
nfft_quotient = 10 #integer setting the quotient or None for default options, this is the fraction of the time series used as lenght of the FFT, 4 is recommended by Schönwiese 2006; if set to None, the default nfft option is used in signal.periodgram(). This option is not used in signal.csd() which always assumes None / default options 
scaling = 'spectrum'
detrend = 'constant' #linear or constant for removing the linear trend or mean only prior to calculating the power spectrum
window = 'hamming' #hamming or hann
repetitions = 1000 #number of repetitions used for randomly reshuffling the time series in order to obtain confidence intervals for random power spectra
ci_percentiles = [2.5,5.,95.,97.5]
cutoff = [] #defines where the x-axis is cut-off in the time-series plots, [] for no cutoff

#visualization options
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'png' #png, pdf, etc.
titlesize = 8. #Font size of the titles in the figures

#execute ###############################################################################################
exec(open('analysis_functions.py').read())

wt_names = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
ref_tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #used to decide whether to calc. the power spectra or not. To this end, continuous time series are needed.

#set <anom> to "no" in any case for yearly aggregations
if aggreg == 'year':
    print('INFO: Calculation of monthly anomaly values does not apply for yearly temporal aggregations; <anom> is thus forced to be "no" in this excursion of the script.')
    anom = 'no'

#get hemispheres for the two target regions
if region1 in ('nh','escena','iberia','eurocordex','medcordex','cordexna','north_atlantic'):
    hemis1 = 'nh'
elif region1 in ('sh','sh_midlats'):
    hemis1 = 'sh'

if region2 in ('nh','escena','iberia','eurocordex','medcordex','cordexna','north_atlantic'):
    hemis2 = 'nh'
elif region2 in ('sh','sh_midlats'):
    hemis2 = 'sh'

for tarwts in tarwts_various:
    wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","").replace(" ","_")
    for tarmonths in tarmonths_various:
        seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','') #needed to plot and save the figures
        yearly_an = np.all(np.isin(ref_tarmonths,tarmonths)) #check whether yearly analysis will be hereafter applied or not
        if yearly_an:
            print('INFO: All target months were requested by the user and the seasonal cycle will thus be plotted.')
        else:
            print('INFO: Seasonal-specific time series were requested by the user and script will be run in reduced mode, i.e. the seasonal cycle is not plotted.')
        for j in list(range(len(model))): 
            print('INFO: loading '+model[j]+', '+mrun[j]+'...')
            #complete the path to the output figures
            periodogram_dir = fig_root+'/'+model[j]+'/'+experiment+'/regional/'+aggreg+'/periodogram/'+wtlabel
            timeseries_dir = fig_root+'/'+model[j]+'/'+experiment+'/regional/'+aggreg+'/timeseries/'+wtlabel
            seascyc_dir = fig_root+'/'+model[j]+'/'+experiment+'/regional/'+aggreg+'/seascyc/'+wtlabel
            #create target directory if missing
            if os.path.isdir(periodogram_dir) != True:
                os.makedirs(periodogram_dir)
            if os.path.isdir(timeseries_dir) != True:
                os.makedirs(timeseries_dir)
            if os.path.isdir(seascyc_dir) != True:
                os.makedirs(seascyc_dir)
                
            if model[j] == 'cera20c':
                timestep = '3h'
                file_startyear = '1901'
                file_endyear = '2010'
            elif model[j] == 'era5':
                timestep = '6h'
                file_startyear = '1979'
                file_endyear = '2020'
            else:
                timestep = '6h'
                file_startyear = '1979'
                file_endyear = '2005'
                
            store_wt = store_wt_orig+'/'+timestep+'/'+experiment
            wt_file_reg1 = store_wt+'/'+hemis1+'/wtseries_'+model[j]+'_'+experiment+'_'+mrun[j]+'_'+hemis1+'_'+file_startyear+'_'+file_endyear+'.nc' #path to the LWT catalogues
            wt_file_reg2 = store_wt+'/'+hemis2+'/wtseries_'+model[j]+'_'+experiment+'_'+mrun[j]+'_'+hemis2+'_'+file_startyear+'_'+file_endyear+'.nc' #path to the LWT catalogues
            
            #aggregate data for each region (reg1, reg2) and sum up for the combined area (reg1reg2)
            if anom == 'no':
                wt_ag_reg1_step,dates_reg1 = load_and_aggregate_wts(wt_file_reg1,region1,taryears,tarwts,aggreg,timestep,latweighting,tarmonths)
                wt_ag_reg2_step,dates_reg2 = load_and_aggregate_wts(wt_file_reg2,region2,taryears,tarwts,aggreg,timestep,latweighting,tarmonths)
                wt_ag_reg1reg2_step = wt_ag_reg1_step + wt_ag_reg2_step
            elif anom == 'yes':
                wt_ag_reg1_step,dates_reg1,seas_cyc_1_step = load_and_aggregate_wts_anom(wt_file_reg1,region1,taryears,tarwts,aggreg,timestep,latweighting,tarmonths,rollw)
                wt_ag_reg2_step,dates_reg2,seas_cyc_2_step = load_and_aggregate_wts_anom(wt_file_reg2,region2,taryears,tarwts,aggreg,timestep,latweighting,tarmonths,rollw)
                wt_ag_reg1reg2_step = wt_ag_reg1_step + wt_ag_reg2_step

            dates_reg1reg2 = dates_reg1.copy()
            
            #check whether the dates are identical
            if any(dates_reg1.isin(dates_reg2) == False):
                raise Exception('ERROR: the dates in <dates_reg1> and <dates_reg2> are not identical!')

            if nfft_quotient == None:
                print('Info: The default nfft options in signal.periodogram() and signal.csd() will be used for (cross) spectral analysis...')
                nfft = None
            elif nfft_quotient > 0:
                nfft = int(round(len(wt_ag_reg1_step.values)/nfft_quotient)) #use optionally in the next 4 lines
                print('Info: As requested by the user, '+str(nfft)+' ffts will be used for spectral analysis and the default option for cross-spectral analysis...')
            else:
                raise Exception('ERROR: unknown entry for <nfft_quotient>!')
            nfft_csd = None #will be always used in signal.csd()
            
            print('INFO: The detrending option applied prior to calculating the power and cross power spectra is '+detrend+'.')
            f_reg1_step, Pxx_reg1_step = signal.periodogram(wt_ag_reg1_step.values, fs=fs, nfft=nfft, window=window, scaling='spectrum', detrend=detrend) #transforms to anomalies prior to calculating frequencies and spectrum by default
            f_reg2_step, Pxx_reg2_step = signal.periodogram(wt_ag_reg2_step.values, fs=fs, nfft=nfft, window=window, scaling='spectrum', detrend=detrend)
            #f_cross_step, Pxx_cross_step = signal.csd(wt_ag_reg1_step.values, wt_ag_reg2_step.values, fs=fs, nfft=nfft, window=window, nperseg=nfft, noverlap=None, detrend=detrend, return_onesided=True, scaling='spectrum', axis=-1, average='mean')
            f_cross_step, Pxx_cross_step = signal.csd(wt_ag_reg1_step.values, wt_ag_reg2_step.values, fs=fs, nfft=nfft_csd, window=window, nperseg=nfft_csd, noverlap=None, detrend=detrend, return_onesided=True, scaling='spectrum', axis=-1, average='mean')
            
            #get confidence intervals from randomly reshuffled time series
            ntime = len(wt_ag_reg1_step)
            #init array to be filled with power spectra from randomly reshuffled time series
            Pxx_reg1_rand = np.zeros((repetitions,len(Pxx_reg1_step)))
            Pxx_reg2_rand = np.copy(Pxx_reg1_rand)
            Pxx_cross_rand = np.zeros((repetitions,len(Pxx_cross_step)))
            print('INFO: Init power spectrum calculations with randomly reshuffled times series and '+str(repetitions)+' repetitions...')
            for rr in list(np.arange(repetitions)):
                rand1 = np.random.randint(0,ntime,ntime)
                rand2 = np.random.randint(0,ntime,ntime)
                f_reg1_rand_step, Pxx_reg1_rand_step = signal.periodogram(wt_ag_reg1_step.values[rand1], fs=fs, nfft=nfft, window=window, scaling='spectrum', detrend=detrend)
                f_reg2_rand_step, Pxx_reg2_rand_step = signal.periodogram(wt_ag_reg2_step.values[rand2], fs=fs, nfft=nfft, window=window, scaling='spectrum', detrend=detrend)
                #f_cross_rand_step, Pxx_cross_rand_step = signal.csd(wt_ag_reg1_step.values[rand1], wt_ag_reg2_step.values[rand2], fs=fs, nfft=nfft, window=window, nperseg=nfft, noverlap=None, detrend=detrend, return_onesided=True, scaling='spectrum', axis=-1, average='mean')
                f_cross_rand_step, Pxx_cross_rand_step = signal.csd(wt_ag_reg1_step.values[rand1], wt_ag_reg2_step.values[rand2], fs=fs, nfft=nfft_csd, window=window, nperseg=nfft_csd, noverlap=None, detrend=detrend, return_onesided=True, scaling='spectrum', axis=-1, average='mean')
                Pxx_reg1_rand[rr,:] = Pxx_reg1_rand_step
                Pxx_reg2_rand[rr,:] = Pxx_reg2_rand_step
                Pxx_cross_rand[rr,:] = Pxx_cross_rand_step
            print('Info: Reshuffling has terminated!')
            #get 95% confidence intervals for the random power spectra
            ci_reg1 = np.percentile(Pxx_reg1_rand,ci_percentiles,axis=0)
            ci_reg2 = np.percentile(Pxx_reg2_rand,ci_percentiles,axis=0)
            ci_cross = np.percentile(Pxx_cross_rand,ci_percentiles,axis=0)

            #plot the loopwise results, spectograms first
            #savename = periodogram_dir+'/'+model[mm]+'_'+str(len(mrun))+'_members_periodogram_LWT_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+city[cc]+'_lon'+str(wt_center.lon.values)+'lat'+str(wt_center.lat.values)+'.'+outformat
            #plot_power_spectrum('single',f,Pxx,ci,fs,window,nfft,detrend,anom,wtnames,tarwts,model[0],'10',taryears,seaslabel,aggreg,city,outformat,dpival,titlesize)

            savename = periodogram_dir+'/periodogram_'+model[j]+'_'+mrun[j]+'_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+'.'+outformat
            titlelabel = 'PS '+wtlabel+' '+model[j].upper()+' '+mrun[j]+' '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg+' nfft'+str(nfft)+' dtrend'+detrend+' anom'+anom+' '+region1
            plot_power_spectrum('single',f_reg1_step,Pxx_reg1_step,ci_reg1,fs,window,nfft,detrend,aggreg,savename,titlelabel,dpival,titlesize)
           
            savename = periodogram_dir+'/periodogram_'+model[j]+'_'+mrun[j]+'_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+region2+'.'+outformat
            titlelabel = 'PS '+wtlabel+' '+model[j].upper()+' '+mrun[j]+' '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg+' nfft'+str(nfft)+' dtrend'+detrend+' anom'+anom+' '+region2
            plot_power_spectrum('single',f_reg2_step,Pxx_reg2_step,ci_reg2,fs,window,nfft,detrend,aggreg,savename,titlelabel,dpival,titlesize)
            
            savename = periodogram_dir+'/cross_spectrum_'+model[j]+'_'+mrun[j]+'_'+wtlabel+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_nfft_'+str(nfft)+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+'_'+region2+'.'+outformat
            titlelabel = 'CPS '+wtlabel+' '+model[j].upper()+' '+mrun[j]+' '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg+' nfft'+str(nfft)+' dtrend'+detrend+' anom'+anom+' '+region1+' '+region2
            plot_power_spectrum('cross',f_cross_step,Pxx_cross_step,ci_cross,fs,window,nfft_csd,detrend,aggreg,savename,titlelabel,dpival,titlesize)

            #then get the seasonal cycle of the aggregated data and plot
            if aggreg != 'year' and anom == 'yes':
                # seas_cyc_1_step =  wt_ag_reg1_step.groupby("time.month").mean("time")
                # seas_cyc_2_step =  wt_ag_reg2_step.groupby("time.month").mean("time")
                # plot_seas_cycle(seas_cyc_1_step,detrend,anom,wtnames,tarwts,model[j],mrun[j],taryears,aggreg,region1,outformat,dpival,titlesize)
                # plot_seas_cycle(seas_cyc_2_step,detrend,anom,wtnames,tarwts,model[j],mrun[j],taryears,aggreg,region2,outformat,dpival,titlesize)
                #init empty array in first passing of the loop
                if j == 0:
                    seas_cyc_1 = np.zeros((len(model),len(seas_cyc_1_step)))
                    seas_cyc_2 = np.copy(seas_cyc_1)
                #fill arrays with loop-wise results
                seas_cyc_1[j,:] = seas_cyc_1_step
                seas_cyc_2[j,:] = seas_cyc_2_step 
            else:
                print('INFO: No seasonal cycles are calculated for aggreg='+aggreg)
            
            #init arrays in the first loop, init is placed here because it depends on the shape of the frequencies and power spectra obtained above
            if j == 0:
                wt_ag_reg1 = np.zeros((len(model),len(wt_ag_reg1_step)))
                wt_ag_reg2 = np.copy(wt_ag_reg1)
                wt_ag_reg1reg2 = np.copy(wt_ag_reg1)    
                f_reg1 = np.zeros((len(model),len(f_reg1_step)))
                f_reg2 = np.copy(f_reg1)
                f_cross = np.zeros((len(model),len(f_cross_step)))
                Pxx_reg1 = np.copy(f_reg1)
                Pxx_reg2 = np.copy(f_reg1)
                Pxx_cross = np.copy(f_cross)
            
            #fill arrays with loop-wise results
            wt_ag_reg1[j,:] = wt_ag_reg1_step.values
            wt_ag_reg2[j,:] = wt_ag_reg2_step.values
            wt_ag_reg1reg2[j,:] = wt_ag_reg1reg2_step.values 
            f_reg1[j,:] = f_reg1_step
            f_reg2[j,:] = f_reg2_step
            f_cross[j,:] = f_cross_step
            Pxx_reg1[j,:] = Pxx_reg1_step
            Pxx_reg2[j,:] = Pxx_reg2_step
            Pxx_cross[j,:] = Pxx_cross_step

        #convert numpy arrays to xarray data arrays
        wt_ag_reg1 = xr.DataArray(data=wt_ag_reg1,coords =[np.arange(len(model)),dates_reg1],dims=['model','time'],name='wtcount')
        wt_ag_reg2 = xr.DataArray(data=wt_ag_reg2,coords =[np.arange(len(model)),dates_reg2],dims=['model','time'],name='wtcount')
        wt_ag_reg1reg2 = xr.DataArray(data=wt_ag_reg1reg2,coords =[np.arange(len(model)),dates_reg1],dims=['model','time'],name='wtcount')

        #get running mean of yearly or otherwise temporally aggregated ensemble mean counts
        means_reg1 =  np.mean(wt_ag_reg1,axis=0)
        runmeans_reg1 = np.convolve(means_reg1, np.ones(meanperiod)/meanperiod, mode='valid')
        means_reg2 =  np.mean(wt_ag_reg2,axis=0)
        runmeans_reg2 = np.convolve(means_reg2, np.ones(meanperiod)/meanperiod, mode='valid')
        means_reg1reg2 =  np.mean(wt_ag_reg1reg2,axis=0)
        runmeans_reg1reg2 = np.convolve(means_reg1reg2, np.ones(meanperiod)/meanperiod, mode='valid')

        ##plot the time series for the temporally aggregated time instances in region 1 and 2, and the sum thereof
        titlelabel = wtlabel+' '+model[j].upper()+' '+str(len(mrun))+'m '+str(taryears[0])+'-'+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtrend '+detrend+' anom '+anom+' '+region1
        savename = timeseries_dir+'/timeseries_'+ensemble_label+'_'+str(len(mrun))+'m_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+'.'+outformat
        plot_time_series(wt_ag_reg1,detrend,aggreg,savename,titlelabel,dpival,titlesize=titlesize)
        
        titlelabel = wtlabel+' '+model[j].upper()+' '+str(len(mrun))+'m '+str(taryears[0])+'-'+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtrend '+detrend+' anom '+anom+' '+region2
        savename = timeseries_dir+'/timeseries_'+ensemble_label+'_'+str(len(mrun))+'m_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtrend_'+detrend+'_anom_'+anom+'_'+region2+'.'+outformat
        plot_time_series(wt_ag_reg2,detrend,aggreg,savename,titlelabel,dpival,titlesize=titlesize)
        
        titlelabel = wtlabel+' '+ensemble_label+' '+str(len(mrun))+'m '+str(taryears[0])+'-'+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtrend '+detrend+' anom '+anom+' '+region1+region2
        savename = timeseries_dir+'/timeseries_'+ensemble_label+'_'+str(len(mrun))+'m_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+region2+'.'+outformat
        plot_time_series(wt_ag_reg1reg2,detrend,aggreg,savename,titlelabel,dpival,titlesize=titlesize)
        
        #plot the seasonal cycles of the two target regions
        if aggreg != 'year' and anom == 'yes' and yearly_an:
            fig = plt.figure()
            timeaxis = np.tile(seas_cyc_1_step.dayofyear.values,(seas_cyc_1.shape[0],1))
            plt.plot(timeaxis.transpose(),seas_cyc_1.transpose(),linestyle='solid')
            plt.plot(timeaxis.transpose(),seas_cyc_2.transpose(),linestyle='dashed')
            #plt.ylim(seas_cyc_1.values.min()-1,seas_cyc_1.values.max()+1)
            #wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")
            plt.title(wtlabel+' '+ensemble_label+' '+str(len(mrun))+' '+str(taryears[0])+'-'+str(taryears[1])+' '+aggreg+' dtrend '+detrend+' anom '+anom+' '+region1+' (solid) and '+region2+' (dashed)',fontsize=titlesize)
            savename = seascyc_dir+'/seasonalcycle_'+ensemble_label+'_'+str(len(mrun))+'m_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+aggreg+'_dtrend_'+detrend+'_anom_'+anom+'_'+str(rollw)+'_'+region1+region2+'.'+outformat
            plt.savefig(savename,dpi=dpival)
            plt.close('all')

    print('Info: get_decvar_regional.py has completed successfully!')
