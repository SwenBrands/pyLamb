# -*- coding: utf-8 -*-

''''This script calcluates long-term trends and power spectra for a range of periods upon reanalysis or GCM data. It reads in various members of a given ensemble,
calculates the member-wise statistics, checks their significance and finally the ensemble's agreement on significance (and for the trends also aggeement on the slope's sign).
The <nfft> parameter used in scipy.signal.periodogram() and welch() refers to the maximal length of the temporal shifts (i.e. to the largest measured period / lowest frequency),
which is the lenght of the time series at the utmost. If set to None, <nfft> is the length of the time series in periodogram() and the length of <nperseg> (see below) in welch().
The <nperseg> input parameter is used in welch() only and defines the length of the overlapping sub-periods taken into account by welch(), where the power spectra are first
calculated seperately for each sub-period and then averaged to get mor robust results. Launch to queue on SMD cluster with e.g.:
qsub -N map_lowfreqvar -l walltime=06:00:00 -l mem=48gb -q himem -e error.log -o out.log -l nodes=1:ppn=8 launchme.sh &

1. Remaining issues:
1.1 Check which are the units for the mapped power spectra ? Is it really <variable unit>² ?
1.2 Save the results (slopes, power spectra etc.) to netCDF format
1.3 The script is ready to apply to alternative reanalyses and GCM; this will be done in the near future

2. Remaining tasks:
2.1 Repeat for GCM ensembles and compare GCM and reanalyses spectra. Do this on the basis of the stored netCDF files from point 1.2 above
2.2 As an alternative to 2.1, compute cross-power spectra (CPS) GCM vs. reanalysis
2.3 To look for forcing agents of low frequency variability, also calcluate CPS for reanalysis and SST indices (among others like e.g. QBO index) 
2.4 Use autocorrelation function in addition to power spectra in order to check the robustness of the results
2.5 Lower the temporal resolution of the analayses from yearly to monthly or weekly accumulations (work in progress)
2.6 Inform spatial average spectral analyses with the maps produced by this script, e.g. define a domain for central Europe in get_decvar_regional.py (this script already fully working for yearly, monthly or weekly LWT counts)
2.7 Use Wavelets ?
'''

#load packages
import numpy as np
import pandas as pd
import xarray as xr
from scipy import fft, arange, signal, stats
import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import os
import time
import pdb
import pymannkendall as mk
from joblib import Parallel, delayed
import time
import gc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#set input parameters
n_par_jobs = 16 #number of parallel jobs, see https://queirozf.com/entries/parallel-for-loops-in-python-examples-with-joblib
region_various = ['nh'] #['nh','sh']
tarwts_various = [[1]] #[[1],[18],[2,10,19],[3,11,20],[4,12,21],[5,13,22],[6,14,23],[7,15,24],[8,16,25],[9,17,26],[27]] # [[18],[7,15,24]], list of lists, loop through various wt combinations for exploratory data analysis; e.g. [5,13,22] are southerly directions, [9,17,26] are northerly ones
tarmonths_various = [[1,2,3,4,5,6,7,8,9,10,11,12]] #[[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3],[4,5,6],[7,8,9],[10,11,12]] #loop through various seasons for exploratory data analysis
taryears = [1901,2010] #start and end yeartaryears = [1979,2005] #start and end year
aggreg = 'year' #temporal aggregation of the 3 or 6-hourly time series: 'year' or '1', '7', '10' or '30' indicating days, must be string format
fig_root = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs' #path to the output figures
store_wt_orig = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'
wtnames = ['PA','DANE','DAE','DASE','DAS','DASW','DAW','DANW','DAN','PDNE','PDE','PDSE','PDS','PDSW','PDW','PDNW','PDN','PC','DCNE','DCE','DCSE','DCS','DCSW','DCW','DCNW','DCN','U']

model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']

#model = ['cera20c','cera20c']
#mrun = ['m0','m1']

#model = ['mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr']
#mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']

#model = ['mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr']
#mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1']

# model = ['era5']
# mrun = ['r1i1p1']
# experiment = 'historical' #historical, 20c, amip, ssp245, ssp585

experiment = '20c' #20c or historical
meanperiod = 10 #used to calculate the rolling mean WT counts for the temporally aggregated data as defined in <aggreg> above, .e.g. 10
anom = 'yes' #remove monthly mean values to obtain anomalies
rollw = 21 #rolling window in days used to calculate the climatological mean which is then removed from each entry of the daily time series; must be an uneven number
testlevel = 0.05 #test-level assumed to decide whether the Mann Kendall tendency is signiificant
mktest = 'original' # original, yue_wang, pre_white, hamed_rao, regional, seasonal, theilslopes (scipy is used in the latter case) Type of Mann Kendall test or modifications thereof described in https://doi.org/10.21105/joss.01556
use_sig = 'mk' #mk or user, chose which significance estimate to use; mk uses the output of the Mann Kendall function stated above, user derives it manually from filtering the p-values with the <testlevel> parameter defined above
relax = 1 #number of members that are allowed to be in disagreement on the sign or significance of the slope if compared to the other members of the ensemble defined in <model> and <mrun>

#visualization options
dpival = 300 #resolution of the output figure in dots per inch
outformat = 'pdf' #png, pdf, etc.
titlesize = 7. #Font size of the titles in the figures
colormap_tr = 'seismic'
colormap_ps = 'hot_r'

#options used for periodgram, not included here so far
fs = 1 #sampling frequency used for calcluating power and cross-power spectra; is applied on temporally aggregated values as defined by <aggreg> above
nfft_quotient = None #must be equal or lower than nperseg_quotient; used to calculate <nfft>, i.e. the length of the maximum temporal shift (equal to the largest period) taken into account in the periodogram. nfft = np.floor(n / nfft_quotient), where n is the sample size of the time series; 4 is recommended by Schönwiese 2006; must be at least 1 or None; if set to None, the default nfft option is used in signal.periodgram() and signal.welch(), implies zero padding when used in Welch (check what this means!). 
nperseg_quotient = 2 #only used by Welch; used to calculate <nperseg>, i.e. the length of the overlapping sub-periods taken into account in Welch, where the power spectra are first calculated seperately for each subperiod and then averaged to get mor robust results in a cross-validation-like manner; nperseg = np.floor(n / nperseg_quotient), where n is the sample size of the time series (for e.g. 50 year superiods of n = 100 years, this quotient is set to 2)
scaling = 'spectrum'
detrend = 'linear' #linear or constant for removing the linear trend or mean only prior to calculating the power spectrum
window = 'hann' #hamming or hann, etc.
repetitions = 1000 #number of repetitions used for randomly reshuffling the time series in order to obtain confidence intervals for random power spectra
ci_percentile = 90.
periodogram_type = 'periodogram' #periodogram or Welch, as implemented in scipy.signal
yearly_units = '%' # count or % (relative frequency); unit of the yearly LWT counts

#execute ###############################################################################################
#check consistency of input setting
if len(model)-relax <= 0:
    raise Exception('ERROR: relax = '+str(relax)+' is too large for ensemble size = '+str(len(model)))

starttime = time.time()
print('INFO: This script will use '+str(n_par_jobs)+' parallel jobs to calculate the long term tendencies and power spectra along all longitudinal grid-boxes at a given latitude.')
exec(open('analysis_functions.py').read())

wt_names = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
ref_tarmonths = [1,2,3,4,5,6,7,8,9,10,11,12] #used to decide whether to calc. the power spectra or not. To this end, continuous time series are needed.

#set <anom> to "no" in any case for yearly aggregations
if aggreg == 'year':
    print('INFO: Calculation of monthly anomaly values does not apply for yearly temporal aggregations; <anom> is thus forced to be "no" in this excursion of the script.')
    anom = 'no'

#loop through regions
for region in region_various:
    #get hemispheres for the target region
    if region in ('nh','escena','iberia','eurocordex','medcordex','cordexna','north_atlantic'):
        hemis = 'nh'
    elif region in ('sh'):
        hemis = 'sh'
    #then loop through requested LWT combinations
    for tarwts in tarwts_various:
        wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")
        for tarmonths in tarmonths_various:
            print('INFO: Calculating '+mktest+' trend and '+periodogram_type+', window '+window+', relax '+str(relax)+' for '+region+' target region, requested WTs '+str(tarwts)+' and requested season '+str(tarmonths)+'...')
            seaslabel = str(tarmonths).replace('[','').replace(']','').replace(', ','') #needed to plot and save the figures
            if seaslabel == '123456789101112':
                seaslabel = 'yearly'

            for mm in list(range(len(model))): 
                print('INFO: loading '+model[mm]+', '+mrun[mm]+'...')
                figs = fig_root+'/'+model[mm] #complete the path to the output figures
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
                    
                store_wt = store_wt_orig+'/'+timestep+'/'+experiment
                wt_file_reg = store_wt+'/'+hemis+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+file_startyear+'_'+file_endyear+'.nc' #path to the LWT catalogues
                nc = xr.open_dataset(wt_file_reg)
                lats = nc.lat.values
                lons = nc.lon.values
                
                #select target years and months
                dates = pd.DatetimeIndex(nc.time.values)
                year_ind = np.where((dates.year >= taryears[0]) & (dates.year <= taryears[1]) & dates.month.isin(tarmonths))[0]
                nc = nc.isel(time=year_ind)
                dates = dates[year_ind]
                
                #get target WTs and set to 1 vs. 0
                arr = xr.zeros_like(nc) #init an xarray Dataset which will be a binary presence-absence array
                tarwt_ind = nc.wtseries.isin(tarwts)
                arr['wtseries'].values[tarwt_ind] = 1
                nc.close()
                del(nc)
                
                #temporal aggregation and optional conversion from yearly counts to yearly relative frequencies
                if aggreg == 'year':
                    if yearly_units == 'count':
                        arr = arr.groupby('time.year').sum('time') #calculate annual mean values
                    elif yearly_units == '%':
                        hours_per_year = arr.groupby('time.year').count()
                        arr = arr.groupby('time.year').sum('time') #calculate annual mean values
                        arr = arr / hours_per_year *100
                    else:
                        raise Exception('ERROR: unknown entry for <yearly_units>!')
                    #month_length = xr_ds.time.dt.days_in_month
                    dates = dates.year
                    yearly_an = np.all(np.isin(ref_tarmonths,tarmonths)) #check whether yearly analysis will be hereafter applied or not
                    if yearly_an:
                        print('INFO: All target months (i.e. yearly LWT counts) were requested by the user and the chance for alising effects is thus reduced.')
                    else:
                        print('WARNING: Seasonal-specific time series were requested by the user and there is a chance for alising effects in the computed power spectra !')
                elif aggreg in ('1','5','7','10','30'): #aggreg in days
                    if timestep == '3h':
                        interval = int(float(aggreg)*8)
                    elif timestep == '6h':
                        interval = int(float(aggreg)*4)
                    else:
                        raise Exception('Error: unknown entry for <timestep> !')
                    arr = arr.rolling(time=interval_f,center=True).sum().dropna('time') #calculate rolling daily mean values
                    arr = arr[0::interval] #remove overlapping data
                    dates = pd.DatetimeIndex(nc.time)
                    print('INFO: the lenght of the time series is '+str(nc.shape[0]))
                else:
                    raise Excpetion('ERROR: unknown entry for <aggreg>!')
                    
                #check consistency for the trend input parameters set by the user
                if (mktest == 'theilslopes') & (use_sig == 'user'):
                    raise Exception('ERROR: mktest = '+mktest+' and use_sig = '+use_sig+' are not compatible!') 
                
                arr_vals = arr.wtseries.values #convert to numpy array
                model_plus_run = [model[xx] + mrun[xx] for xx in np.arange(len(model))]
                if mm == 0:
                    #the varying fft is currently only used in signal.periodogram() below; for signal.welch() it is set to None and thus eguals the length of nperseg
                    if periodogram_type == 'Welch':
                        #nperseg = int(np.floor(arr_vals.shape[0]/nperseg_quotient))+1 #only used if <periodogram_type> = 'Welch', get the lenght (e.g. in year) of the sub-sequents used to calculate the FFTs
                        nperseg = int(np.floor(arr_vals.shape[0]/nperseg_quotient)) #only used if <periodogram_type> = 'Welch', get the lenght (e.g. in year) of the sub-sequents used to calculate the FFTs
                        nfft = None #i.e. the lenght of nperseg will be used in welch()
                    elif periodogram_type == 'periodogram':
                        nperseg = None
                        if nfft_quotient == None:
                            nfft = None
                        elif nfft_quotient >= 1:
                            nfft = int(np.floor(arr_vals.shape[0]/nfft_quotient))
                        elif int(nfft_quotient) < 1:
                            raise Exception('ERROR: <nfft_quotient> was set too small! It must be set at least to 1 in order to nfft <= n, where n is the length of the time series')
                        else:
                            raise Exception('ERROR: unknown entry for <nfft_quotient>!')
                    else:
                        raise Exception('ERROR: check entry for <periodogram_type>!')
                    print('INFO: nfft and nperseg are set to '+str(nfft)+' and '+str(nperseg)+', respectively.')
                        
                    #get number of frequencies for a test sample to get the corresponding dimension in the arrays to be initialized
                    if periodogram_type == 'periodogram':
                        f_def, Pxx_def = signal.periodogram(arr_vals[:,0,0], fs=fs, nfft=nfft, window=window, scaling=scaling, detrend=detrend)
                    elif periodogram_type == 'Welch': #see https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.signal.welch.html
                        f_def, Pxx_def = signal.welch(arr_vals[:,0,0], fs=fs, nfft=nfft, nperseg=nperseg, noverlap=None, window=window, scaling=scaling, detrend=detrend)
                    else:
                        raise Exception('ERROR: Check entry for <periodogram_type>!')
                        
                    print('INFO: init xr DataArrays for the slope, intercept, p-value and significance True/False...')
                    slope = np.zeros((len(model),arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    pval = np.zeros((len(model),arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    intercept = np.zeros((len(model),arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    sig_def = np.zeros((len(model),arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    print('INFO: numpy arrays for storing the results from spectral analysis are initialized...')
                    n_freq = len(f_def) #number of frequecies / periods at which power spectra are calcualted
                    fq =  np.zeros((len(model),n_freq,arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    Pxx =  np.zeros((len(model),n_freq,arr.wtseries.shape[1],arr.wtseries.shape[2]))
                    ci =  np.zeros((len(model),n_freq,arr.wtseries.shape[1],arr.wtseries.shape[2]))                    
                
                #run a parallel loop along all longitudinal grid-boxes of a given latitude
                for jj in np.arange(arr_vals.shape[2]):
                    time_series = arr_vals[:,:,jj]
                    par_result = Parallel(n_jobs=n_par_jobs)(delayed(get_lowfreq_var)(time_series[:,ii],mktest,testlevel,periodogram_type,nfft,fs,window,scaling,detrend,repetitions,ci_percentile,nperseg=nperseg) for ii in np.arange(arr_vals.shape[1]))
                    for ii in np.arange(len(par_result)):                   
                        ##fill the np arrays with the results of each loop through mm (models), ii (lons) and jj (lats)
                        #for the trends
                        slope[mm,ii,jj] = par_result[ii][0].slope
                        intercept[mm,ii,jj] = par_result[ii][0].intercept
                        pval[mm,ii,jj] = par_result[ii][0].p
                        sig_def[mm,ii,jj] = par_result[ii][0].h
                        #for the power spectra
                        fq[mm,:,ii,jj] = par_result[ii][1]
                        Pxx[mm,:,ii,jj] = par_result[ii][2]
                        ci[mm,:,ii,jj] = par_result[ii][3]

                ##free memory
                del(par_result)
            
            #get mask indicating grid-boxes where the trend has the same sign and is significant in len(model)-relax ensemble members
            signif_tr = np.copy(pval) # tr for "trend"
            sigmask_tr = pval < testlevel #signifcant trend
            spurmask_tr = pval >= testlevel #spurious trend
            signif_tr[sigmask_tr] = 1
            signif_tr[spurmask_tr] = 0
            if use_sig == 'user':
                sumsignif_tr = np.sum(signif_tr,axis=0)
            elif use_sig == 'mk':
                sumsignif_tr = np.sum(sig_def,axis=0)
            else:
                raise Exception('ERROR: check entry for <use_sig>!')
            
            slope_sum = np.copy(slope)
            posmask = slope_sum > 0
            negmask = slope_sum <= 0
            slope_sum[posmask] = 1
            slope_sum[negmask] = -1
            slope_sum = np.sum(slope_sum,axis=0)
            agree_ind_tr = (np.abs(slope_sum) >= (len(model)-relax)) & (sumsignif_tr >= (len(model)-relax))
            
            ##get mask indicating grid-boxes where the power spectra for a given period are significant in len(model)-relax ensemble members, critical values above which significance is assumed were obtained above from random reshuffles of the original time series
            period = 1/fq
            period[np.isinf(period)] = np.nan
            signif_ps = np.copy(Pxx)
            sigmask_ps = Pxx > ci
            spurmask_ps = Pxx <= ci
            signif_ps[sigmask_ps] = 1
            signif_ps[spurmask_ps] = 0
            sumsignif_ps = np.sum(signif_ps,axis=0)
            agree_ind_ps = sumsignif_ps >= len(model)-relax

            #set geographical variables used in both kind of maps (trends and power spectra)
            xx,yy = np.meshgrid(lons,lats)
            xx[xx > 180] = xx[xx > 180]-360
            xx = np.transpose(xx)
            yy = np.transpose(yy)
            runlabel = str(len(mrun)) #number of runs
            modellabel = model[mm] #last model
            halfres = np.abs(np.diff(lats))[0]/2
            
            #plot trend agreement map
            maxval_tr = np.max(np.abs(slope))
            minval_tr = maxval_tr*-1
            
            if region == 'nh':
                map_proj = ccrs.NorthPolarStereo() #ccrs.PlateCarree()
                #map_proj = ccrs.PlateCarree() #ccrs.PlateCarree()
            elif region == 'sh':
                map_proj = ccrs.SouthPolarStereo()
                #map_proj = ccrs.PlateCarree()
            else:
                raise Exception('ERROR: check entry for <region>!')
            
            pattern_tr = np.mean(slope,axis=0)
            title_tr = mktest+' '+wtlabel+' '+modellabel.upper()+' '+runlabel+'m relax'+str(relax)+' '+str(taryears[0])+'-'+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtr '+detrend+' an '+anom+' '+region
            
            #make output directory if it does not exist
            wtlabel2save = wtlabel.replace(' ','_')
            path_trend = figs+'/regional/'+aggreg+'/maps/'+region+'/'+seaslabel+'/trend/'+wtlabel2save
            if os.path.isdir(path_trend) != True:
                os.makedirs(path_trend)            
            savename_tr = path_trend+'/'+mktest+'_'+modellabel+'_'+runlabel+'m_relax'+str(relax)+'_'+wtlabel2save+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtr_'+detrend+'_an_'+anom+'_'+region+'.'+outformat
            get_map_lowfreq_var(pattern_tr,xx,yy,agree_ind_tr,minval_tr,maxval_tr,dpival,title_tr,savename_tr,halfres,colormap_tr,titlesize,yearly_units)
            
            #plot power spectrum agreement maps for each period
            period_unique = period[0,:,0,0]
            print('INFO: Power spectra have been calcualted for the following '+str(len(period_unique))+' periods:')
            print(period_unique)
            Pxx_med = np.median(Pxx,axis=0)
            maxval_ps = np.max(np.abs(Pxx_med))
            minval_ps = 0
            
            #make output directory if it does not exist
            path_periodogram = figs+'/regional/'+aggreg+'/maps/'+region+'/'+seaslabel+'/periodogram/'+wtlabel2save
            if os.path.isdir(path_periodogram) != True:
                os.makedirs(path_periodogram)
                
            for pp in np.arange(len(period_unique)):
                if agree_ind_ps[pp,:,:].flatten().sum()==0:
                    print('INFO: No ensemble agreement as defined by the user (relax: '+str(relax)+') was found for the '+str(np.round(period_unique[pp],1))+' years period. Proceed to the next period....')
                    continue
                
                pattern_ps = Pxx_med[pp,:,:]
                cbar_label = 'Power spectrum amplitude ('+yearly_units+'$²$)'
                #add nperseg in title and filename in case Welch periodogram is used
                if periodogram_type == 'Welch':
                    title_ps = periodogram_type+' '+str(np.round(period_unique[pp],1))+'y '+window+' nfft'+str(nfft)+' nseg'+str(nperseg)+' ci'+str(round(ci_percentile))+' '+wtlabel+' '+modellabel.upper()+' '+runlabel+'m relax'+str(relax)+' '+str(taryears[0])+' '+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtr '+detrend+' an '+anom+' '+region
                    savename_ps = path_periodogram+'/'+periodogram_type+'_'+str(np.round(period_unique[pp],1))+'y_'+window+'_nfft'+str(nfft)+'_nseg'+str(nperseg)+'_ci'+str(round(ci_percentile))+'_'+modellabel+'_'+runlabel+'m_relax'+str(relax)+'_'+wtlabel2save+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtr_'+detrend+'_an_'+anom+'_'+region+'.'+outformat
                elif periodogram_type == 'periodogram':
                    title_ps = periodogram_type+' '+str(np.round(period_unique[pp],1))+'y '+window+' nfft'+str(nfft)+' ci'+str(round(ci_percentile))+' '+wtlabel+' '+modellabel.upper()+' '+runlabel+'m relax'+str(relax)+' '+str(taryears[0])+' '+str(taryears[1])+'-'+seaslabel+' '+aggreg+' dtr '+detrend+' an '+anom+' '+region
                    savename_ps = path_periodogram+'/'+periodogram_type+'_'+str(np.round(period_unique[pp],1))+'y_'+window+'_nfft'+str(nfft)+'_ci'+str(round(ci_percentile))+'_'+modellabel+'_'+runlabel+'m_relax'+str(relax)+'_'+wtlabel2save+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg+'_dtr_'+detrend+'_an_'+anom+'_'+region+'.'+outformat
                else:
                    raise Exception('ERROR: check entry for <periodogram_type>!')
                    
                get_map_lowfreq_var(pattern_ps,xx,yy,agree_ind_ps[pp,:,:],minval_ps,maxval_ps,dpival,title_ps,savename_ps,halfres,colormap_ps,titlesize,cbar_label,origpoint=None)
            gc.collect() #explicetly free memory

endtime = time.time()
elaptime = endtime - starttime
print('INFO: map_lowfreq_var.py has ended successfully! The elapsed time is '+str(elaptime)+'seconds, exiting now...')
quit()
