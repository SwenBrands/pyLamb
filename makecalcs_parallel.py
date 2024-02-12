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
exec(open('lambtyping_parallel.py').read())
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)
exec(open('analysis_functions.py').read()) #a function assigning metadata to the models in <model> (see below)

########################################################################
#Note: The time period is filtered in the previous interpolation step accomplished by <interpolator_xesmf.py>

n_par_jobs = 16 #number of parallel jobs, see https://queirozf.com/entries/parallel-for-loops-in-python-examples-with-joblib
experiment = 'historical' #historical, 20c, amip, ssp245, ssp585, piControl or dcppA
home = os.getenv('HOME')
filesystem = 'lustre'
hemis = 'nh' #sh or nh
saveindices = 'no' #save the 6 indices of the Lamb scheme, 'yes' or 'no'
lead_time = 1 #lead time in years, currently only used for dcppA experiments

# historical runs extended with ssp245 to compare with dcppA runs below
model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
mrun = ['r1i1p1f1','r4i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r21i1p1f1']

# # dcppA runs
# model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
# mrun = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']

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

## accomplished CERA-20C LWT catalogues for both the NH and SH
#model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
#mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']

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
    taryears, timestep = get_target_period(model[mm],experiment,cmip_f=cmip,lead_time_f=lead_time)
    
    #Print the main script configuration to inform the user during execution from one GCM / reanalysis to another
    print('INFO: Calculating Lamb Weather types for '+model[mm]+' for '+experiment+', time period '+str(taryears[0])+' to '+str(taryears[1])+', and lead-time (applied only for dcppA experiment) of '+str(lead_time)+' year(s)...')
    
    #create target directory if necessary
    tarpath_step = tarpath + '/' + timestep + '/' + experiment + '/' + hemis
    if os.path.isdir(tarpath_step) != True:
        os.makedirs(tarpath_step)

    # set path to archive
    if experiment == 'dcppA':
        archivo = srcpath + '/' + model[mm] + '/'+timestep+'/' + experiment + '/' + mrun[mm] + '/psl_interp' + '/psl_interp_'+timestep+'rPlev_' + model[mm] + '_' + experiment + '_' + mrun[mm] + '_' + hemis + '_' + str(lead_time)+'y.nc'
    elif experiment in ('historical', '20c', 'amip', 'ssp245', 'ssp585', 'piControl'):
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
        par_result = Parallel(n_jobs=n_par_jobs)(delayed(lambtyping_parallel)(dataset, lons_lamb_0, lats_lamb_0, tarres, i, j, hemis) for i in list(range(glon_dim)))
        #par_result = Parallel(n_jobs=n_par_jobs)(delayed(lambtyping_parallel)(dataset, lons_lamb_0, lats_lamb_0, tarres, i, j, hemis) for j in list(range(glat_dim)))
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
    if experiment == 'dcppA':
        newfile = tarpath_step+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(lead_time_from_nc[0:2])+'y_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'
    else:
        newfile = tarpath_step+'/wtseries_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+str(taryears[0])+'_'+str(taryears[1])+'.nc'

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
    print('INFO: saving results file at '+newfile)
    #outnc.to_netcdf(newfile)
    outnc.to_netcdf(newfile, encoding = {'wtseries': {'dtype': 'int16'}})
    outnc.close()
    del(outnc,newfile)
    
    #optionally save circulation indices
    if saveindices == 'yes':
        print('INFO: saving categorial data and the 6 circulation indices....')
        circinds =  ['w','s','zw','zs','z','f']
        circnames =  ['wflow','sflow','wvort','svort','vort','flow']
        fullnames =  ['westerly flow','southerly flow','westerly shear vorticity','southerly shear vorticity','total shear vorticity','resultant flow'] #following Jones et al. 1993
        for ci in list(range(len(circinds))):
            if experiment == 'dcppA':
                newfile = tarpath_step +'/'+circinds[ci]+'_'+model[mm]+'_'+experiment+'_'+mrun[mm]+'_'+hemis+'_'+'_'+str(lead_time_from_nc[0:2])+'y_'+taryears[0]+'_'+taryears[1]+'.nc'
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
    elif saveindices == 'no':
        print('INFO: saving  categorial data only....')
    else:
        raise Exception('ERROR: check entry for <saveindices>!')
    endtime2 = time.time()
    elaptime2 = endtime2 - starttime2
    print('The elapsed time for processing '+model[mm]+' is '+str(elaptime2)+'seconds, proceeding to the next model now..')

endtime = time.time()
elaptime = endtime - starttime
print('makecalcs_parallel.py has ended successfully! The elapsed time is '+str(elaptime)+'seconds, exiting now...')
quit()

