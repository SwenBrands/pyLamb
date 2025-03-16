# -*- coding: utf-8 -*-

'''Short version history:
1. <calendar> list to be set as input parameter in previous versions
is now directly read from the file containing the interpolated psl data
2. <mapping> option is no longer supported because there the separate
script <get_composites.py> does this job more efficiently
3. This is the parallel version of makecalcs_py3.py
4. Calculation of the intermediate Lamb indices <w>, <s>, <zw>, <zs>,
<z> and <f> is commented below in order to limit RAM use.
Uncomment lines 159-164 when more memory is available.
@author: Swen Brands, swen.brands@gmail.com
'''

import pandas as pd
import numpy as np
import netCDF4 as nc4
import matplotlib.pylab as plt
import xarray as xr
import datetime
from pdb import set_trace as bp
from mpl_toolkits.basemap import Basemap
import os
from math import sqrt
from joblib import Parallel, delayed
import time
import gc
import psutil
import pdb #then type <pdb.set_trace()> at a given line in the code below
exec(open('lambtyping_parallel.py').read())
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)
exec(open('analysis_functions.py').read()) #a function assigning metadata to the models in <model> (see below)

########################################################################
#Note: The time period is filtered in the previous interpolation step accomplished by <interpolator_xesmf.py>

n_par_jobs = 32 #number of parallel jobs, see https://queirozf.com/entries/parallel-for-loops-in-python-examples-with-joblib
experiment = '20c' #historical, 20c, amip, ssp245, ssp585, rcp85 piControl or dcppA; currently 20 for cera20c and historical for era5; fix this inconsistency in the future !
home = os.getenv('HOME')
filesystem = 'lustre'
hemis = 'nh' #sh or nh
save_indices = 'no' #save the 6 indices of the Lamb scheme, 'yes' or 'no'
calc_monthly_counts = 'yes' # yes or no, calculate monthly LWT count, this part runs in parallel
monthly_calc = 'serial' # serial or parallel, how to calculate the monthly LWT counts
verbose_level = 2 #detail of verbose level used by the joblib Parallel function
compression_level = None #integer between 1 and 9 or None, compression level of the output files
lead_time = 1 #lead time in years, currently only used for dcppA experiments
lead_time_concept = 'LT' #lead time concept: FY or LT; FY to consider forecast years (starting on January 1 and ending in December 31) or LT to leave the forecast as is, i.e. starting on November 1 and ending in October 31 of the following year for EC-Earth r1i1pf1f to r10i1pf1f.
force_6h = 'no' #currently only applied for ERA5 and CERA-20C; if set to yes, filters out the data at 0, 6, 12, 18 UTC in case the input data is 3-hourly.

#save monthly counts for all 27 types
tarwts = np.arange(1,28) #The full LWT scheme will be considered, 1 = PA, 27 = U
tarwts_name = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN','PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC','DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']

# #save monthly counts for 11 types
# tarwts = [[1],[2,10,19],[3,11,20],[4,12,21],[5,13,22],[6,14,23],[7,15,24],[8,16,25],[9,17,26],[18],[27]] #list of lists containing target LWTs for which monthly frequencies will be stored
# #tarwts_name = ['PA', 'DANE_PDNE_DCNE', 'DAE_PDE_DCE', 'DASE_PDSE_DCSE', 'DAS_PDS_DCS', 'DASW_PDSW_DCSW', 'DAW_PDW_DCW', 'DANW_PDNW_DCNW', 'DAN_PDN_DCN','PC','U'] #original names for 11 types
# tarwts_name = ['PA','NE','E','SE','S','SW','W','NW','N','PC','U'] #summarized names for 11 types

# # NH and SH catalogues calculated from CESM2-LE historical runs
# mrun = ['1001.001','1021.002','1041.003','1061.004','1081.005','1101.006','1121.007','1141.008','1161.009','1181.010','1231.001','1231.002','1231.003','1231.004','1231.005','1231.006','1231.007','1231.008','1231.009','1231.010','1251.001','1251.002','1251.003','1251.004','1251.005','1251.006','1251.007','1251.008','1251.009','1251.010','1281.001','1281.002','1281.003','1281.004','1281.005','1281.006','1281.007','1281.008','1281.009','1281.010','1301.001','1301.002','1301.003','1301.004','1301.005','1301.006','1301.007','1301.008','1301.009','1301.010']
# model = ['cesm2']*len(mrun)

# # ERA5
# model = ['era5']
# mrun =  ['r1i1p1']

# # historical runs extended with ssp245 to compare with dcppA runs below
# model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
# mrun = ['r1i1p1f1','r4i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r21i1p1f1']

# # dcppA runs
# model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
# mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']

# # ## accomplished rcp85 and ssp585 LWT catalogues for both the NH and SH
# model = ['miroc_esm']
# mrun =  ['r1i1p1']

# ## accomplished historical LWT catalogues for both the NH and SH
# model = ['ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','noresm2_mm','awi_esm_1_1_lr','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','hadgem2_cc','hadgem2_es','hadgem3_gc31_mm','noresm2_lm','noresm2_lm','nesm3','nesm3','mri_esm2_0','noresm2_mm','miroc_es2l','miroc_es2l','ec_earth3_veg','ec_earth3_veg_lr','miroc6','ec_earth3_aerchem','ec_earth3_cc','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','noresm2_lm','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','hadgem2_es','mpi_esm_1_2_lr','access_esm1_5','noresm2_mm','ipsl_cm5a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','mpi_esm_1_2_hr','ipsl_cm5a_lr','ipsl_cm5a_lr','ipsl_cm5a_lr','ipsl_cm5a_lr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mri_esm2_0','mri_esm2_0','mri_esm2_0','fgoals_g2','fgoals_g3','kiost_esm','iitm_esm','taiesm1','csiro_mk3_6_0','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','gfdl_cm3','giss_e2_r','era5','kace_1_0_g','cmcc_cm2_hr4','inm_cm5','canesm5','gfdl_esm2g','miroc6','ec_earth3_veg','gfdl_esm4','bcc_csm1_1','cnrm_cm6_1','cnrm_cm6_1','cmcc_esm2','ipsl_cm5a_lr','interim','jra55','cnrm_cm6_1_hr','cnrm_cm6_1','ec_earth3_aerchem','ec_earth3_cc','cnrm_esm2_1','giss_e2_1_g', 'sam0_unicon', 'bcc_csm2_mr', 'gfdl_cm4','ec_earth3','access13', 'mpi_esm_mr', 'cmcc_cm','access10', 'ccsm4', 'ec_earth', 'canesm2', 'mpi_esm_lr', 'cnrm_cm5', 'giss_e2_h', 'inm_cm4', 'miroc_esm', 'mri_esm1', 'noresm1_m', 'ipsl_cm5a_mr', 'miroc5', 'hadgem2_es','mri_esm2_0','mpi_esm_1_2_ham','mpi_esm_1_2_lr','mpi_esm_1_2_lr','cnrm_esm2_1','access_cm2','access_esm1_5','cmcc_cm2_sr5','ipsl_cm6a_lr','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','nesm3','nesm3','nesm3']
# mrun =  ['r5i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1','r11i1p1f1','r2i1p1f1','r1i1p1f1','r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r1i1p1','r1i1p1','r1i1p1f3','r1i1p1f1','r2i1p1f1','r1i1p1f1','r5i1p1f1','r4i1p1f1','r1i1p1f1','r1i1p1f2','r5i1p1f2','r1i1p1f1','r1i1p1f1','r3i1p1f1','r1i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r9i1p1f1','r10i1p1f1','r3i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r22i1p1f1','r15i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1','r32i1p1f1','r2i1p1','r8i1p1f1','r3i1p1f1','r3i1p1f1','r4i1p1','r10i1p1f1','r11i1p1f1','r12i1p1f1','r13i1p1f1','r10i1p1f1','r2i1p1','r3i1p1','r5i1p1','r6i1p1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r2i1p1f1','r3i1p1f1','r5i1p1f1','r1i1p1','r3i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1','r5i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r1i1p1','r6i1p1','r1i1p1','r1i1p1f1','r1i1p1f1','r2i1p1f1','r1i1p2f1','r1i1p1','r1i1p1f1','r6i1p1f1','r1i1p1f1','r1i1p1','r2i1p1f2','r3i1p1f2','r1i1p1f1','r1i1p1','r1i1p1','r1i1p1','r1i1p1f2','r1i1p1f2','r1i1p1f1','r1i1p1f1','r1i1p1f2','r1i1p1f1','r1i1p1f1', 'r1i1p1f1', 'r1i1p1f1', 'r24i1p1f1','r1i1p1', 'r1i1p1', 'r1i1p1','r1i1p1', 'r6i1p1', 'r12i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r6i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f2','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r23i1p1f1','r25i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1']

## accomlished piControl catalogues for both the NH and SH
#model = ['mpi_esm_1_2_lr']
#mrun =  ['r1i1p1f1']

## accomplished amip LWT catalogues for both the NH and SH
#model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6']
#mrun =  ['r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r9i1p1f1','r10i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']

##missing years in historical experiment for this run, e.g. 1877, 1880, 1888, 1892, 1905, 1910
#model = ['ec_earth3_veg']
#mrun = ['r10i1p1f1']

# accomplished CERA-20C LWT catalogues for both the NH and SH
model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']

###EXECUTE ####################################################################
starttime = time.time()
print('INFO: This script will use '+str(n_par_jobs)+' parallel jobs to calculate the LWTs along all longitudinal grid-boxes at a given latitude.')

#root to input netcdf files
if filesystem == 'extdisk': #for external hard disk
    srcpath = '/media/swen/ext_disk2/datos/GCMData/'
    tarpath = '/media/swen/disk2/datos/lamb_cmip5/results_v2'
    rundir = home+'/datos/tareas_meteogalicia/lamb_cmip5/pyLamb'
elif filesystem == 'lustre': #for filesystems at MeteoGalicia mounted by Sergio
    srcpath = '/lustre/gmeteo/WORK/swen/datos/GCMData'
    tarpath = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2'
    rundir = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb'
else:
    raise Exception('ERROR: unknown entry for <filesystem>!')
os.chdir(rundir)

#define the Lamb Coordinate system
lons_lamb_0 = np.array([-5, 5, -15, -5, 5, 15, -15, -5, 5, 15, -15, -5, 5, 15, -5, 5])
lats_lamb_0 = np.array([40, 40, 35, 35, 35, 35, 30, 30, 30, 30, 25, 25, 25, 25, 20, 20])
if hemis == 'sh':
    lats_lamb_0 = np.flipud(lats_lamb_0)*-1

tarres=2.5
halfres=tarres/2

#loop through the distinct gcms and reanalyses
for mm in list(range(len(model))):
    starttime2 = time.time()
    #get metadata for this GCM
    runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm])

    #define the time period the GCM data is interpolated for as a function of the experiment and considered GCM
    taryears, timestep = get_target_period(model[mm],experiment,cmip_f=cmip,lead_time_f=lead_time,lead_time_concept_f=lead_time_concept)
    
    #Print the main script configuration to inform the user during execution from one GCM / reanalysis to another
    print('INFO: Calculating Lamb Weather types for '+model[mm]+' for '+experiment+', time period '+str(taryears[0])+' to '+str(taryears[1])+', and lead-time (applied only for dcppA experiment) of '+str(lead_time)+' year(s) with lead-time concept '+lead_time_concept)
    
    # set path to input files
    if experiment == 'dcppA':
        # archivo = srcpath + '/' + model[mm] + '/'+timestep+'/' + experiment + '/' + mrun[mm] + '/psl_interp' + '/psl_interp_'+timestep+'rPlev_' + model[mm] + '_' + experiment + '_' + mrun[mm] + '_' + hemis + '_' + str(lead_time)+'y.nc'
        archivo = srcpath + '/' + model[mm] + '/'+timestep+'/' + experiment + '/' + mrun[mm] + '/psl_interp' + '/psl_interp_'+timestep+'rPlev_' + model[mm] + '_' + experiment + '_' + mrun[mm] + '_' + hemis + '_' + lead_time_concept+'_'+ str(lead_time)+'y.nc'
    elif experiment in ('historical', '20c', 'amip', 'ssp245', 'ssp585', 'rcp85', 'piControl'):
        archivo = srcpath + '/' + model[mm] + '/'+timestep+'/' + experiment + '/' + mrun[mm] + '/psl_interp' + '/psl_interp_'+timestep+'rPlev_' + model[mm] + '_' + experiment + '_' + mrun[mm] + '_'+hemis+'.nc'
    else:
        raise Exception('ERROR: unknown entry for <experiment> input parameter!')
    
    #set latitudinal and longitudinal limits
    if hemis == 'nh':
        lat_bounds = [30, 70]
    elif hemis == 'sh':
        lat_bounds = [-70, -30]
    else:
        raise Exception('ERROR: check entry for <hemis>')
    lon_bounds = [0,357.5]
    #lon_bounds = [355,357.5] #use for short tests

    print('INFO: loading '+archivo)
    dataset = xr.open_dataset(archivo) #open the dataset
    dates = pd.DatetimeIndex(dataset.time.values)
    yearinds = np.squeeze(np.where((dates.year >= int(taryears[0])) & (dates.year <= int(taryears[-1]))))
    dataset = dataset.isel(time=yearinds)
    dates = dates[yearinds]
    print('INFO: Loading the whole psl data array into memory... this needs large amounts of memory if the time vector in the input file is long (i.e. many years are to be loaded), but significantly speeds up the grid-box data selection in lambtyping_parallel.py!')
    dataset.psl.values #load psl data array into memory
    
    #load metadata
    attrs_glob = dataset.attrs.items()
    attrs_psl = dataset.psl.attrs.items()
    attrs_lon = dataset.lon.attrs.items()
    attrs_lat = dataset.lat.attrs.items()
    attrs_time = dataset.time.attrs.items()
    calendar_from_nc = dataset.calendar
    calendar_metadata_from_nc = dataset.calendar
    #optionally get the lead-time from input file and check whether the file metadata is consistent which what is requested in this script (see lead_time input parameter above)
    if experiment == 'dcppA':
        lead_time_from_nc = dataset.lead_time
        lead_time_from_nc = lead_time_from_nc[0:2].replace(' ','',) #replace has been placed to get rid of the empty space in case lead_time_from_nc = '1 ' 
        if lead_time_from_nc == str(lead_time):
            print('The lead-time retrieved from the file '+archivo+' corresponds to the lead-time requested by the user, i.e. '+str(lead_time)+' year(s)')
        else:
            raise Exception('ERROR ! The lead-time stored in '+archivo+' is '+lead_time_from_nc+' and does not correspond to the lead-time requested by the user in the lead_time input parameter, which is '+str(lead_time))

    #operate in a grid box (resolution = tarres)
    dimt = len(dates)
    glat_dim = int((lat_bounds[1] - lat_bounds[0])/tarres) + 1
    glon_dim = int((lon_bounds[1] - lon_bounds[0])/tarres) + 1
    #define central value
    center_0 = np.array(((lons_lamb_0[1] + lons_lamb_0[0]) / 2., lats_lamb_0[6]))
    center_lons = np.zeros(glon_dim)
    center_lats = np.zeros(glat_dim)
    lons_lamb = np.copy(lons_lamb_0)
    lats_lamb = np.copy(lats_lamb_0)
    
    #init output arrays
    wtseries = np.zeros((dimt, glon_dim, glat_dim))
    #optionally init numpy arrays for the 6 Lamb circulation indices
    if save_indices == 'yes':
        print('As requested by the user, the 6 Lamb circulation indices w, s, zw, zs, z and f will be calculated alongside the types themselves...')
        w = np.copy(wtseries)
        s = np.copy(wtseries)
        zw = np.copy(wtseries)
        zs = np.copy(wtseries)
        z = np.copy(wtseries)
        f = np.copy(wtseries)
    
    #par_result = Parallel(n_jobs=n_par_jobs)(delayed(lambtyping_parallel)(dataset, lons_lamb_0, lats_lamb_0, tarres, i, j, hemis) for i,j in zip(list(range(glon_dim)),list(range(glat_dim))))
    for j in list(range(glat_dim)):
    #for i in list(range(glon_dim)):
        center_lats[j] = lats_lamb[6]
        
        # ##parallel part explicitly freeing workers after using Parallel, see https://stackoverflow.com/questions/67495271/joblib-parallel-doesnt-terminate-processes
        # current_process = psutil.Process()
        # subproc_before = set([p.pid for p in current_process.children(recursive=True)])
        par_result = Parallel(n_jobs=n_par_jobs,verbose=verbose_level)(delayed(lambtyping_parallel)(dataset, lons_lamb_0, lats_lamb_0, tarres, i, j, hemis) for i in list(range(glon_dim)))
        # subproc_after = set([p.pid for p in current_process.children(recursive=True)])
        # for subproc in subproc_after - subproc_before:
            # print('Killing process with pid {}'.format(subproc))
            # psutil.Process(subproc).terminate()

        # parallel = Parallel(n_jobs=n_par_jobs,verbose=verbose_level)
        # par_result = parallel(delayed(lambtyping_parallel)(dataset, lons_lamb_0, lats_lamb_0, tarres, i, j, hemis) for i in list(range(glon_dim)))
        # parallel.terminate()

        #assign output
        for i in list(range(len(par_result))):
            wtseries[:,i,j] = par_result[i][0]
            wtnames = par_result[i][1]
            wtcode = par_result[i][2]
            #w[:,i,j] = par_result[i][3]
            #s[:,i,j] = par_result[i][4]
            #zw[:,i,j] = par_result[i][5]
            #zs[:,i,j] = par_result[i][6]
            #z[:,i,j] = par_result[i][7]
            #f[:,i,j] = par_result[i][8]
            SLP = par_result[i][9]
            lon = par_result[i][10]
            lat = par_result[i][11]
            dirdeg = par_result[i][12]
            center_lons[i] = par_result[i][13]
            #print('Center latitude is '+str(par_result[i][14]))
            center_lats[j] = par_result[i][14] #center_lats is equal for each i (longitude at a given latitude)
        del(par_result)
        gc.collect() #explicetly free memory

    #try to obtain interpolation method from input file
    try:
        regridding_method = dataset.attrs['xesmf_regridding_method']
    except:
        print("INFO: no interpolation method was saved in "+archivo)
    dataset.close()
    
    #save file for each model and output variable
    wtseries = wtseries.astype(np.int16) #convert to integer values

    outnc = xr.DataArray(wtseries, coords=[dates, center_lons, center_lats], dims=['time', 'lon', 'lat'], name='wtseries')
    outnc.attrs['long_name'] = 'Lamb Weather Types'
    outnc.attrs['standard_name'] = 'Lamb Weather Types'
    #outnc.attrs['wtnames'] = str(wtnames)
    outnc.attrs['wtnames'] = np.array(wtnames)
    outnc.attrs['wtcode'] = np.int8(wtcode)
    outnc.attrs['units'] = 'categorical'
    outnc.attrs['reference'] = 'doi: 10.5194/gmd-15-1375-2022'
    outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
    outnc.attrs['info'] = 'Attributes with prefix <udata> (= underlying data) refer to metadata originally contained in the sea-level pressure source files obtained from the ESGF or reanalysis providers (e.g. Copernicus or JMA), which has been copied to this file.'
    outnc.attrs['calendar'] = calendar_from_nc
    outnc.attrs['calendar_metadata'] = calendar_metadata_from_nc
    outnc.attrs['Conventions'] = 'CF-1.6'
    #optionally add the lead-time of the forecast (for initialized GCM experiments only)
    if experiment == 'dcppA':
        outnc.attrs['lead_time'] = lead_time_from_nc
        outnc.attrs['lead_time_concept'] = lead_time_concept
    for item in attrs_glob:
        outnc.attrs['udata_'+item[0]] = item[1]
    for item in attrs_psl:
        outnc.attrs['udata_'+item[0]] = item[1]
    for item in attrs_time:
        outnc.time.attrs['udata_'+item[0]] = item[1]
        outnc.time.attrs['long_name'] = 'time'
        outnc.time.attrs['standard_name'] = 'time'
    for item in attrs_lon:
        outnc.lon.attrs['udata_'+item[0]] = item[1]
        outnc.lon.attrs['long_name'] = 'longitude'
        outnc.lon.attrs['standard_name'] = 'longitude'
        outnc.lon.attrs['units'] = 'degrees_east'
    for item in attrs_lat:
        outnc.lat.attrs['udata_'+item[0]] = item[1]
        outnc.lat.attrs['long_name'] = 'latitude'
        outnc.lat.attrs['standard_name'] = 'latitude'
        outnc.lat.attrs['units'] = 'degrees_north'

    #try to save interpolation method in output file    
    try:
        outnc.attrs['xesmf_regridding_method'] = regridding_method
    except:
        print("INFO: interpolation method cannot be saved because it is not available from "+archivo)
    
    #filter out 6-hourly values for reanalysis data after doing some checks
    accum_hours = np.unique(outnc.time.dt.hour)

    #force 6-hourly output data, irrespective of the entry in the get_target_period() function
    if force_6h == 'yes':
        if model[mm] not in ('era5','cera20c'):
            raise ValueError('The force_6h input parameter is currently only used for model[m] set to era5 or cera20c !')
        if len(accum_hours) != 8:
            raise ValueError('The accum_hours variable is expected to have a lenght of 8, but has not !')
        hour_ind = np.isin(outnc.time.dt.hour,[0,6,12,18])
        outnc = outnc.isel(time=hour_ind)
        accum_hours = np.unique(outnc.time.dt.hour) # overwrite accum_hours with new values
        timestep = '6h' #overwrite timestep

    accum_label = str(accum_hours).replace('[','').replace(']','') # accumulation label used below as attribute in monthly output dataset
    
    #check consistency between accumulation hours and timestep
    if (timestep == '6h' and len(accum_hours) != 4) or (timestep == '3h' and len(accum_hours) != 8):
        raise ValueError('timestep and accum_hours variables are not consistent !')

    #set output directory name and create it on disk, if necessary
    tarpath_step = tarpath + '/' + timestep + '/' + experiment + '/' + hemis
    tarpath_step_mon = tarpath + '/mon/' + experiment + '/' + hemis #to save monthly LWT frequencies for internal use
    
    if os.path.isdir(tarpath_step) != True:
        os.makedirs(tarpath_step)
    if os.path.isdir(tarpath_step_mon) != True:
        os.makedirs(tarpath_step_mon)
    
    #set path and name of the output files
    if experiment == 'dcppA':
        newfile = tarpath_step+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+lead_time_concept+'_'+str(lead_time_from_nc[0:2])+'y_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'
        newfile_mon = tarpath_step_mon+'/wtcount_mon_'+timestep+'_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+lead_time_concept+'_'+str(lead_time_from_nc[0:2])+'y_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'
    else:
        newfile = tarpath_step+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'
        newfile_mon = tarpath_step_mon+'/wtcount_mon_'+timestep+'_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'
    
    #get monthly frequencies for all LWTs or combinations thereof defined in tarwts; for internal use only - no metadata is attached to the xarray dataset
    if calc_monthly_counts == 'yes':
        print('Start aggregation of monthly LWT counts...')
        nr_months = len(outnc.resample(time='1M'))
        #init output numpy array with dimensions LWTs x time (month) x lon x lat
        arr_tarwts_np = np.zeros((len(tarwts),nr_months,outnc.shape[1],outnc.shape[2]))
        
        if monthly_calc == 'parallel':
            print('WARNING: The parallel monthly LWT count calculation is unstable for unkown reasons. The workers seemingly do to close correctly and the script crashes after several iterations!')
            
            # #parallel part explicitly freeing workers after using Parallel, see https://stackoverflow.com/questions/67495271/joblib-parallel-doesnt-terminate-processes
            # current_process = psutil.Process()
            # subproc_before = set([p.pid for p in current_process.children(recursive=True)])
            par_result = Parallel(n_jobs=n_par_jobs,verbose=verbose_level)(delayed(get_monthly_lwt_counts)(outnc,tarwts[lwt]) for lwt in np.arange(len(tarwts)))
            # subproc_after = set([p.pid for p in current_process.children(recursive=True)])
            # for subproc in subproc_after - subproc_before:
                # print('Killing process with pid {}'.format(subproc))
                # psutil.Process(subproc).terminate()

            gc.collect() #explicetly free memory
            for lwt in np.arange(len(par_result)):
                arr_tarwts_np[lwt,:,:,:] = par_result[lwt][0].values
            monthly_dates = pd.DatetimeIndex(par_result[0][0].time)
            days_per_month = par_result[0][1]
            del(par_result)
        elif monthly_calc == 'serial':
            for lwt in np.arange(len(tarwts)):
                #modify origional LWT time series containing 27 types to a binary absence (0) - occurrence (1) time series of the requested types only
                bin_array = np.zeros(outnc.shape)
                tarwt_ind = np.where(outnc.isin(tarwts[lwt]))
                bin_array[tarwt_ind] = 1
                arr_tarwts = xr.DataArray(data=bin_array,coords=[pd.DatetimeIndex(outnc.time),outnc.lon,outnc.lat],dims=['time','lon','lat'],name='binary_lwt')
                #calculate monthly sums and assign
                arr_tarwts = arr_tarwts.resample(time='1M').sum()
                monthly_dates = pd.DatetimeIndex(arr_tarwts.time)
                arr_tarwts_np[lwt,:,:,:] = arr_tarwts.values
            days_per_month = arr_tarwts.time.dt.days_in_month
            if any(days_per_month < 28):
                raise Exception('ERROR: unexpected entry for <days_per_month> !')
        else:
            raise Exception('ERROR: check entry for <monthly_calc> input parameter !')

        arr_tarwts = xr.DataArray(data=arr_tarwts_np,coords=[tarwts_name,monthly_dates,outnc.lon,outnc.lat],dims=['lwt','time','lon','lat'],name='wtcount').astype('int16')
        days_per_month = xr.DataArray(data=days_per_month,coords=[pd.DatetimeIndex(arr_tarwts.time)],dims=['time'],name='days_per_month').astype('int16')
        out_dataset_mon = xr.Dataset({'counts': arr_tarwts, 'days_per_month': days_per_month})
        #set metadata of the monthly dataset
        out_dataset_mon.attrs['accumulation_hours'] = accum_label
        
        #save monthly counts in netcdf format and close the respective Python objects
        if isinstance(compression_level,int):
            print('Saving monthly LWT counts at '+newfile_mon+' with compression level '+str(compression_level)+'...')
            out_dataset_mon.to_netcdf(newfile_mon, encoding = {'counts': {'dtype':'int16', 'zlib':True, 'complevel':compression_level},'days_per_month': {'dtype': 'int16'}})
        elif compression_level == None:
            print('Saving monthly LWT counts at '+newfile_mon+' without compression...')
            out_dataset_mon.to_netcdf(newfile_mon)
        else:
            raise Exception('ERROR: check entry for <compression_level> input parameter !')
        out_dataset_mon.close()
        arr_tarwts.close()
        days_per_month.close()
        del(out_dataset_mon,newfile_mon,arr_tarwts,days_per_month)
        print('The aggregation to monthly LWT counts has been accomplished and stored!')
    
    elif calc_monthly_counts == 'no':
        print('On request of the user, monthly LWT counts are not calculated !')
    else:
        raise Exception('ERROR: check entry for <calc_monthly_counts> input parameter !')
    
    #then store the instantaneous and monthly files and close all related xr objectss
    print('Saving hourly instantaneous LWT time series at '+newfile)
    outnc.to_netcdf(newfile, encoding = {'wtseries': {'dtype':'int16', 'zlib':True, 'complevel':compression_level}})
    outnc.close()
    del(outnc,newfile)
    
    #optionally save circulation indices
    if save_indices == 'yes':
        print('INFO: saving categorial data and the 6 circulation indices....')
        circinds =  ['w','s','zw','zs','z','f']
        circnames =  ['wflow','sflow','wvort','svort','vort','flow']
        fullnames =  ['westerly flow','southerly flow','westerly shear vorticity','southerly shear vorticity','total shear vorticity','resultant flow'] #following Jones et al. 1993
        for ci in list(range(len(circinds))):
            if experiment == 'dcppA':
                newfile = tarpath_step +'/'+circinds[ci]+'_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+lead_time_concept+'_'+str(lead_time_from_nc[0:2])+'y_'+taryears[0]+'_'+taryears[1]+'.nc'
            else:
                newfile = tarpath_step +'/'+circinds[ci]+'_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+taryears[0]+'_'+taryears[1]+'.nc'
            outnc = xr.DataArray(eval(circinds[ci]), coords=[dates, center_lons, center_lats], dims=['time', 'lon', 'lat'], name=circnames[ci])
            outnc.attrs['long_name'] = str(fullnames[ci])
            outnc.attrs['standard_name'] = str(circinds[ci])
            outnc.attrs['name'] = str(fullnames[ci])
            outnc.attrs['units'] = 'hPa per 10 degrees latitude'
            outnc.attrs['reference'] = 'doi: 10.5194/gmd-15-1375-2022'
            outnc.attrs['contact'] = 'Swen Brands, swen.brands@gmail.com'
            outnc.attrs['info'] = 'Attributes with prefix <udata> (= underlying data) refer to the metadata originally contained in the sea-level pressure source files obtained from the ESGF or reanalysis providers (e.g. Copernicus or JMA), which has been copied to this file.'
            outnc.attrs['calendar'] = calendar_from_nc
            outnc.attrs['calendar_metadata'] = calendar_metadata_from_nc
            outnc.attrs['Conventions'] = 'CF-1.6'
            #optionally add the lead-time of the forecast (for initialized GCM experiments only)
            if experiment == 'dcppA':
                outnc.attrs['lead_time'] = lead_time_from_nc
                outnc.attrs['lead_time_concept'] = lead_time_concept
            for item in attrs_glob:
                outnc.attrs['udata_'+item[0]] = item[1]
            for item in attrs_psl:
                outnc.attrs['udata_'+item[0]] = item[1]
            for item in attrs_time:
                outnc.time.attrs['udata_'+item[0]] = item[1]
                outnc.time.attrs['long_name'] = 'time'
                outnc.time.attrs['standard_name'] = 'time'

            for item in attrs_lon:
                outnc.lon.attrs['udata_'+item[0]] = item[1]
                outnc.lon.attrs['long_name'] = 'longitude'
                outnc.lon.attrs['standard_name'] = 'longitude'
                outnc.lon.attrs['units'] = 'degrees_east'
            for item in attrs_lat:
                outnc.lat.attrs['udata_'+item[0]] = item[1]
                outnc.lat.attrs['udata_'+item[0]] = item[1]
                outnc.lat.attrs['long_name'] = 'latitude'
                outnc.lat.attrs['standard_name'] = 'latitude'
                outnc.lat.attrs['units'] = 'degrees_north'
            
            #try to save interpolation method in output file
            try:
                outnc.attrs['basemap.interpoption'] = regridding_method
            except:
                print("INFO: interpolation method cannot be saved because it is not available from "+archivo)
            outnc.to_netcdf(newfile)
            outnc.close()
            del(outnc,newfile)
    elif save_indices == 'no':
        print('INFO: saving  categorial data only....')
    else:
        raise Exception('ERROR: check entry for <save_indices>!')

    endtime2 = time.time()
    elaptime2 = endtime2 - starttime2
    print('The elapsed time for processing '+model[mm]+' is '+str(elaptime2)+'seconds, proceeding to the next model now..')

endtime = time.time()
elaptime = endtime - starttime
print('makecalcs_parallel.py has ended successfully! The elapsed time is '+str(elaptime)+'seconds, exiting now...')
quit()

