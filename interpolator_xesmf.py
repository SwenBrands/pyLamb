# -*- coding: utf-8 -*-
"""
This script cuts out reanalysis data from ERA-Interim and JRA-55 and GCM data from ESGF (downloaded with wget scripts before) and interpolates them 
to a regular lat-lon grid with a resolution of <tarres> degrees using bilenear interpolation. The user can decide on whether to do this for the northern and southern hemisphere by setting
<mode> to 'nh' or 'sh'. For very high resolution GCMs, a reduced precision of 'int32' is sufficient for the SLP data in Pa this script was written for.

Short version history:
1. <calendar> list to be set as input parameter in previous versions
is now directly read from the file containing the interpolated psl data
2. Files are now loaded one by one to save memory
3. Defalt basemap interpolation option is 3 (cubic spline) in this
version to reduce small-scale variablity in high-resolution GCMs and 
reanalysis datasets.
see https://docs.xarray.dev/en/stable/user-guide/weather-climate.html
@author: Swen Brands, swen.brands@gmail.com
4. use of xarray.convert_calendar to bring all non-standard calendars to standard calendars
5. replace basemap interpolation by xesmf interpolation using an own, clean environment.
Being in the shell, type <conda activate xesmf_env> to activate and <conda deactivate> to
 deactivate this environment. Parallel computing was removed in this version because xesmf
is very quick.
"""
import pandas as pd
import numpy as np
#import netCDF4 as nc4 #netcdf4 installation is necessary for xarray to work with the "netcdf4" engine, but it is not necessary to explicitely import the module.
import xarray as xr
import datetime
import dask
import sys
import os
import time
from cftime import DatetimeNoLeap
import xesmf as xe
exec(open('get_historical_metadata.py').read()) #a function assigning metadata to the models in <model> (see below)
exec(open('analysis_functions.py').read()) #a function assigning metadata to the models in <model> (see below)

##MEMORY EFFICIENT VERSION OF interpolator_north.py, loads GCM data file by file using xarray.open_dataset instead of xarray.open_mfdataset
tarres=2.5
precision = 'int32' #normally float32, int32 for cnrm_cm6_1 models and cnrm_cm5 test case and generally for highres models, for cnrm_cm6_1_hr it only works if started from the bash prompt (i.e. not within ipython)
experiment = 'amip' #historical, 20c, amip, piControl, ssp245, ssp585
regridding_method = 'patch' #bilinear
filesystem = 'lustre' #<lustre> or <extdisk>, used to select the correct path to the source netCDF files
hemis = 'nh'
printfilesize = 'no' #print memory size of the data array subject to interpolation from the individual input netCDF files in the source directory. Depending on the GCM's resolution and the number of years stored in the file, this is most memory greedy object of the script and may lead to a kill of the process.
home = os.getenv('HOME')

## historical runs successfully interpolated for the NH and SH
#model = ['ec_earth3_veg','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','noresm2_mm','awi_esm_1_1_lr','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','hadgem2_cc','hadgem2_es','hadgem3_gc31_mm','noresm2_lm','noresm2_lm','nesm3','nesm3','mri_esm2_0','noresm2_mm','miroc_es2l','miroc_es2l','ec_earth3_veg','ec_earth3_veg_lr','miroc6','ec_earth3_aerchem','ec_earth3_cc','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','mpi_esm_1_2_lr','noresm2_lm','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','hadgem2_es','mpi_esm_1_2_lr','access_esm1_5','noresm2_mm','ipsl_cm5a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','ipsl_cm6a_lr','mpi_esm_1_2_hr','ipsl_cm5a_lr','ipsl_cm5a_lr','ipsl_cm5a_lr','ipsl_cm5a_lr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mri_esm2_0','mri_esm2_0','mri_esm2_0','fgoals_g2','fgoals_g3','kiost_esm','iitm_esm','taiesm1','csiro_mk3_6_0','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','gfdl_cm3','giss_e2_r','era5','kace_1_0_g','cmcc_cm2_hr4','inm_cm5','canesm5','gfdl_esm2g','miroc6','ec_earth3_veg','gfdl_esm4','bcc_csm1_1','cnrm_cm6_1','cnrm_cm6_1','cmcc_esm2','ipsl_cm5a_lr','interim','jra55','cnrm_cm6_1_hr','cnrm_cm6_1','ec_earth3_aerchem','ec_earth3_cc','cnrm_esm2_1','mpi_esm_1_2_hr','giss_e2_1_g', 'sam0_unicon', 'bcc_csm2_mr', 'gfdl_cm4','ec_earth3','access13', 'mpi_esm_mr', 'cmcc_cm','access10', 'ccsm4', 'ec_earth', 'canesm2', 'mpi_esm_lr', 'cnrm_cm5', 'giss_e2_h', 'inm_cm4', 'miroc_esm', 'mri_esm1', 'noresm1_m', 'ipsl_cm5a_mr', 'miroc5', 'hadgem2_es','mri_esm2_0','mpi_esm_1_2_ham','mpi_esm_1_2_lr','mpi_esm_1_2_lr','cnrm_esm2_1','access_cm2','access_esm1_5','cmcc_cm2_sr5','ipsl_cm6a_lr','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','nesm3','nesm3','nesm3']
#mrun =  ['r5i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r6i1p1f1','r11i1p1f1','r2i1p1f1','r1i1p1f1','r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r1i1p1','r1i1p1','r1i1p1f3','r1i1p1f1','r2i1p1f1','r1i1p1f1','r5i1p1f1','r4i1p1f1','r1i1p1f1','r1i1p1f2','r5i1p1f2','r1i1p1f1','r1i1p1f1','r3i1p1f1','r1i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r9i1p1f1','r10i1p1f1','r3i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r22i1p1f1','r15i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1','r32i1p1f1','r2i1p1','r8i1p1f1','r3i1p1f1','r3i1p1f1','r4i1p1','r10i1p1f1','r11i1p1f1','r12i1p1f1','r13i1p1f1','r10i1p1f1','r2i1p1','r3i1p1','r5i1p1','r6i1p1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r2i1p1f1','r3i1p1f1','r5i1p1f1','r1i1p1','r3i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1','r5i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r1i1p1','r6i1p1','r1i1p1','r1i1p1f1','r1i1p1f1','r2i1p1f1','r1i1p2f1','r1i1p1','r1i1p1f1','r6i1p1f1','r1i1p1f1','r1i1p1','r2i1p1f2','r3i1p1f2','r1i1p1f1','r1i1p1','r1i1p1','r1i1p1','r1i1p1f2','r1i1p1f2','r1i1p1f1','r1i1p1f1','r1i1p1f2','r1i1p1f1','r1i1p1f1','r1i1p1f1', 'r1i1p1f1', 'r1i1p1f1', 'r24i1p1f1','r1i1p1', 'r1i1p1', 'r1i1p1','r1i1p1', 'r6i1p1', 'r12i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r6i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1', 'r1i1p1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f2','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r23i1p1f1','r25i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1']
#mycalendar = ['gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','noleap','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','360_day','360_day','360_day','noleap','noleap','gregorian','gregorian','gregorian','noleap','gregorian','gregorian','gregorian','gregorian','proleptic_gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','noleap','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','360_day','gregorian','gregorian','noleap','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','noleap','noleap','noleap','gregorian','noleap','noleap','gregorian','gregorian','gregorian','gregorian','gregorian','noleap','noleap','gregorian','360_day','noleap','noleap','noleap','noleap','gregorian','gregorian','noleap','noleap','gregorian','gregorian','noleap','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','noleap', 'noleap', 'noleap', 'noleap','gregorian','proleptic_gregorian', 'gregorian', 'gregorian','proleptic_gregorian', 'noleap', 'gregorian', 'noleap', 'gregorian', 'gregorian', 'noleap', 'noleap', 'gregorian', 'gregorian', 'noleap', 'gregorian', '360_day', '360_day','gregorian','gregorian','gregorian','gregorian','gregorian','proleptic_gregorian','proleptic_gregorian', 'noleap','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian']

## amip runs successfully interpolated for the NH and SH
#model = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
#mrun = ['r9i1p1f1','r10i1p1f1','r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1','r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r9i1p1f1','r10i1p1f1']
#mycalendar = ['gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian']

model = ['ec_earth3','ec_earth3']
mrun = ['r9i1p1f1','r10i1p1f1']
mycalendar = ['gregorian','gregorian']

## piControl runs successfully interpolated for the NH and SH
#model = ['mpi_esm_1_2_lr']
#mrun = ['r1i1p1f1']
#mycalendar = ['gregorian']

##years are lacking in this simulation:
#model = ['ec_earth3_veg']
#mrun = ['r10i1p1f1']
#mycalendar = ['gregorian']

#model = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
#mrun = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']
#mycalendar = ['gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian','gregorian']

###EXECUTE ####################################################################
starttime = time.time()

#root to input netcdf files
if filesystem == 'extdisk': #for external hard disk
    root1 = '/media/swen/ext_disk2/datos/GCMData/'
    rundir = home+'/datos/tareas/lamb_cmip5/pyLamb'
elif filesystem == 'lustre': #for filesystems at MeteoGalicia mounted by Sergio
    root1 = '/lustre/gmeteo/WORK/swen/datos/GCMData/'
    rundir = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/pyLamb'
else:
    raise Exception('ERROR: unknown entry for <filesystem>!')
os.chdir(rundir)

print('INFO: This script is run in the working directory '+rundir)
print('INFO: all interpolations are for the '+hemis.upper()+'!')
print('INFO: all interpolations are for '+experiment.upper()+' experiments!')
print('INFO: precision was set to '+precision+'!')
print('INFO: xesmf regridding method is: '+regridding_method)

calendar_list = [] #will be filled with calendars for all GCMs
#set name for lat and lon coordinates, and data variable containing the SLP values within the input netCDF file. Note that the time coordinate is assumed to have the name "time" in any case.
for mm in list(range(len(model))):
    #get metadata for this GCM
    runspec,complexity,family,cmip,rgb,marker,latres_atm,lonres_atm,lev_atm,latres_oc,lonres_oc,lev_oc,ecs,tcr = get_historical_metadata(model[mm])
    
    #define the time period the GCM data is interpolated for as a function of the experiment and considered GCM
    taryears, timestep = get_target_period(model[mm],experiment,cmip)
    
    #Print the main script configuration to inform the user during execution from one GCM to another
    print('INFO: Interpolating '+model[mm]+' for '+experiment+' and time period '+str(taryears[0])+' to '+str(taryears[1])+'...')

    #obtain numpy vector containing all target years
    years = np.arange(taryears[0],taryears[-1]+1)
    
    if model[mm] in ('interim','era5'):
        latname = 'latitude' #'latitude'
        lonname = 'longitude' #'longitude'
        varname = 'msl'
        timestep = '6h'
    elif model[mm] == 'ec_earth3' and mrun[mm] == 'r2i1p1f1':
        latname = 'latitude'
        lonname = 'longitude'
        varname = 'psl'
        timestep = '6h'
    elif model[mm] == 'cera20c':
        latname = 'latitude' #'latitude'
        lonname = 'longitude' #'longitude'
        varname = 'msl'
        timestep = '3h'
    else:
        latname = 'lat'
        lonname = 'lon'
        varname = 'psl'
        timestep = '6h'
    
    root2 = '/'+timestep+'/'+experiment+'/'#second part of root to source nc files
    
    #check whether the target directory where the interpolated psl values will be save exists. If does not exists, create it.
    savedir = root1 + model[mm] + root2 + mrun[mm] + '/psl_interp'
    if os.path.isdir(savedir) != True:
        os.makedirs(savedir)

    #clean previously generated temporary files
    prev_tmpdir = root1 + model[mm] + root2 + mrun[mm]+'/psl_interp'
    listdir_tmpfiles = os.listdir(prev_tmpdir)
    if  len(listdir_tmpfiles) > 0:
        dropind = []
        for ff in list(range(len(listdir_tmpfiles))):
            if listdir_tmpfiles[ff][0:3] == 'psl':
                dropind.append(ff)
        listdir_tmpfiles = np.delete(listdir_tmpfiles,dropind).tolist()
        if  len(listdir_tmpfiles) > 0:
            print('WARNING: Previously generated temporary files have been detected in '+prev_tmpdir)
            print('INFO: Deleting previously generated temporary files:')
            print(listdir_tmpfiles)
            os.chdir(prev_tmpdir)
            for tt in listdir_tmpfiles:
                os.remove(tt)        
            del(tt)
            os.chdir(rundir)
        else:
            print('INFO: Previously generated files have been detected in '+prev_tmpdir+', but these are not temporary files of the type tmp...nc generated by this script. Therefore, they are not deleted.')
        del(prev_tmpdir,listdir_tmpfiles,dropind,ff)
    else:
        print('INFO: No previously generated temporary files have been detected in '+prev_tmpdir)
        del(listdir_tmpfiles)

    #generate a list of nc files in source directory containing the GCM data
    listdir_raw = os.listdir(root1 + model[mm] + root2 + mrun[mm])
    dropind = []
    for ff in list(range(len(listdir_raw))):
        if listdir_raw[ff][-3:] != '.nc' and listdir_raw[ff][-4:] != '.nc4':
            dropind.append(ff)
    listdir = np.delete(listdir_raw,dropind).tolist()
    #put listdir into alphabetical order, note that the characters of the filenames must be lowercase, see https://www.tutorialspoint.com/How-to-list-down-all-the-files-alphabetically-using-Python
    listdir = sorted(listdir)
    #listdir = listdir[0:2] #uncomment to check the script with two input files only; two are needed because the loop starting in line 141 assumes a string if listdir only contains a single file.
    ##print results
    print('INFO: The following files and directories are located in the source directory:')
    print(listdir_raw)
    print('INFO: The following list of files will be loaded:')
    print(listdir)
    
    tmpfiles_list = []
    for infile in listdir:
        fullpath = root1 + model[mm] + root2 + mrun[mm]+'/'+infile
        print('INFO: loading dataset '+fullpath)
        dataset = xr.open_dataset(fullpath, engine='netcdf4')
        
        #bring lat, lon and variable names to "lat", "lon" and "psl"
        if model[mm] in ('interim','era5','cera20c'):
            dataset = dataset.rename({lonname: 'lon', latname: 'lat'})
            dataset['psl'] = dataset[varname]
            dataset = dataset.drop(varname)
        
        #from hereon, the dimensions of the dataset are "time", "lat", "lon"; the variable name is "psl"
        attrs_glob = dataset.attrs.items()
        attrs_psl = dataset['psl'].attrs.items()
        attrs_lon = dataset['lon'].attrs.items()
        attrs_lat = dataset['lat'].attrs.items()
        attrs_time = dataset.time.attrs.items()

        #retain only NH or SH
        #lonbounds_c = [ -15. , 377.5 ]
        lonbounds_c = [ -15. , 375. ]
        lats_step = dataset.variables['lat'][:].values
        if hemis == 'nh':
            hemind = np.where(lats_step >=0)[0]
            latbounds = [16., 86.5]
            latbounds_c = [17.5 , 82.5]
        elif hemis == 'sh':
            hemind = np.where(lats_step <=0)[0]
            latbounds = [-86.5,-16]
            latbounds_c = [-82.5,-17.5]
        else:
            raise Exception('ERROR: check entry for <hemis>')

        dataset = dataset.isel(lat=hemind)
        dates = dataset.indexes['time'] #returns a CFTimeIndex

        #retain 00, 06, 12 and 18 data for 3-hourly data from CanESM5
        if model[mm] == 'canesm5':
            print('INFO: only hours 1:30, 7:30, 13:30 and 19:30 are retained from '+model[mm]+ ' dataset...')
            ind6h = np.where((dates.hour == 1) | (dates.hour == 7) | (dates.hour == 13) | (dates.hour == 19))[0]
            dataset = dataset.isel(time = ind6h)
            #it is necessary to reload the dates in this case
            del(dates)
            dates = dataset.indexes['time'] #returns a CFTimeIndex
            
        #set the calendar
        try:
            calendar_from_nc = dataset.indexes['time'].calendar
            calendar_metadata = 'The calendar information was taken from the source netCDF file.'
            print('INFO: calendar is taken from the source netCDF file!')
        except:
            calendar_from_nc = mycalendar[mm]
            calendar_metadata = 'The calendar information was set by Swen Brands after revising the time dimension in the source netCDF file.'
            print('Exception: No calendar is found in the source netCDF file. It is thus set manually from the <mycalendar> list defined at the beginning of this script!')
            
        print('INFO: The calendar for '+model[mm]+' is '+str(calendar_from_nc)+'.')

        #treat lons and lats
        lons = dataset.variables['lon'][:].values
        lats = dataset.variables['lat'][:].values
        #make index with additional longitudes at the eastern and western boundaries        
        tarind1=np.squeeze(np.array(np.where((lons >= -180))))
        tarind2=np.squeeze(np.array(np.where((lons >= 344) & (lons < 360))))
        tarind3=np.squeeze(np.array(np.where((lons >= 0) & (lons < 16))))
        lon_inds = np.concatenate((np.concatenate((tarind2, tarind1), axis=0), tarind3), axis=0)
        lat_inds = np.squeeze(np.array(np.where((lats >= latbounds[0]) & (lats <= latbounds[1]))))
    
        if model[mm] in ('interim','era5','jra55','cera20c'):
            lat_inds = np.flipud(lat_inds)
    
        lats = lats[lat_inds]
        lonsinterp = np.concatenate((np.concatenate((lons[tarind2]-360, lons[tarind1]), axis=0), lons[tarind3]+360), axis=0)
        lons = np.concatenate((np.concatenate((lons[tarind2], lons[tarind1]), axis=0), lons[tarind3]), axis=0)
    
        #load year-specific subset and drop unnecessary variables
        yearinds = np.squeeze(np.where((dates.year >= years[0]) & (dates.year <= years[-1])))
        if  yearinds.size == 0:
            print('WARNING: The temporal coverage of the input file '+fullpath+' is completely out of range of the years solicited in <taryears> above. Thus, we will continue to the next file !')
            dataset.close()
            del(dataset,lons,lon_inds,lonsinterp,lats,lat_inds,yearinds,tarind1,tarind2,tarind3,dates)
            continue
        
        #exception for miroc6, having one single value at 01-01-1979 and the rest before in the first nc file
        if (yearinds.size == 1) & (dates.year[-1] == 1979) & (model[mm] in ('miroc6','fgoals_g2','cnrm_cm5','miroc_es2l','awi_esm_1_1_lr','hadgem3_gc31_mm')):
            print('INFO: Exception is inlcuded for '+model[mm]+', and year '+str(dates.year[-1])+' with a single timestamp on 1979-01-01 OO:00:00: The index number for the prior date is included to have two indices for loading the nc file with xr.load_dataset(). This dummy index deleted afterwards.')
            yearinds = np.array((yearinds-1,int(yearinds)))
            dates = dates[yearinds]
        else:
            dates = dates[yearinds]

        dataset = dataset.isel(time=yearinds, lat=lat_inds, lon=lon_inds)
        
        print('The header of the joint nc file is as follows....')
        print(dataset)
        if model[mm] in ('cnrm_cm6_1','cnrm_cm6_1_hr'): 
            print('Dropping variable <time_bounds> for '+model[mm]+' and '+mrun[mm]+'...')
            dataset = dataset.drop(labels='time_bounds')
        elif model[mm] == 'canesm5': #exception for 3-hourly psl data from CanESM5
            print('Dropping variable <time_bnds>, <lat_bnds> and <lon_bnds> for '+model[mm]+' and '+mrun[mm]+'...')
            dataset = dataset.drop(labels=['time_bnds','lat_bnds','lon_bnds'])
        else:
            print('No variable is dropped...')

        psl = dataset['psl']

        #print the size of the array
        if printfilesize == 'yes':
            print('info: the size of the input dataset is '+str(sys.getsizeof(psl.values)))
        
        dataset.close() #close netcdf file
        
        #Generate meshgrid
        [xx,yy] = np.meshgrid(lonsinterp,lats)
        #then define the coarse grid and interpolate to it
        lonsc = np.linspace(lonbounds_c[0],lonbounds_c[1],int(np.abs((lonbounds_c[0]-lonbounds_c[1]))/tarres+1))
        latsc = np.linspace(latbounds_c[0],latbounds_c[1],int(np.abs((latbounds_c[0]-latbounds_c[1]))/tarres+1))
        [xxc,yyc] = np.meshgrid(lonsc,latsc)
        #exception for miroc6
        if yearinds.size == 1:
            dimt = yearinds.size
        else:
            dimt = len(dates)
            
        dimlon = len(lonsc)
        dimlat = len(latsc)
        #psl = psl.to_dataset()
        psl = psl.transpose('time','lat','lon')
        
        #and interpolate
        print('INFO: starting interpolation for '+fullpath)
        ds_out = xr.Dataset({"lat": (["lat"], latsc, {"units": "degrees_north"}),"lon": (["lon"], lonsc, {"units": "degrees_east"}),})
        regridder = xe.Regridder(psl, ds_out, regridding_method)
        pslc = regridder(psl)

        pslc = pslc.transpose('time','lon','lat')
        pslc['time'] = dates
        pslc.name = 'psl'
        
        #create xarray dataset and save to a temporary netcdf file; one for each input nc file
        starthour = str(dates[0]).replace('-','').replace(' ','').replace(':','')
        endhour = str(dates[-1]).replace('-','').replace(' ','').replace(':','')
        tmpfile = root1 + model[mm] + root2 + mrun[mm] + '/psl_interp' + '/tmp_interp_'+timestep+'rPlev_' + model[mm] +'_'+experiment+'_' + mrun[mm] + '_' + hemis +'_'+starthour+'_'+endhour+'.nc'
        print('INFO: creating temporary file '+tmpfile)

        #convert non-standard calendars, filter time period and save in netCDF format
        if calendar_from_nc in ('360_day','noleap','julian'):
            print('INFO: converting '+dates.calendar+' to standard calendar...')
            pslc = pslc.convert_calendar('standard', dim='time', align_on='year')
        datesconvert = pslc.indexes['time']
        getind  = np.where((datesconvert.year >= taryears[0]) & (datesconvert.year <= taryears[-1]))[0]
        pslc = pslc.isel(time=getind)
        pslc = pslc.astype(precision)
        
        pslc.to_netcdf(tmpfile)
        
        ##close files and append name of the temporary files to a list to remove them once all of them are written and joined to a single one (see below).
        psl.close()
        pslc.close()
        ds_out.close()        
        tmpfiles_list.append(tmpfile)
        ##free memory by deleting all variables within this loop
        del(dataset,psl,pslc,ds_out,regridder,tmpfile,starthour,endhour,xx,yy,xxc,yyc,lons,lonsc,lon_inds,lonsinterp,lats,latsc,lat_inds,yearinds,tarind1,tarind2,tarind3,dates)
        
    #then load the newly created temporary files and save as one
    print('INFO: concatanate from year '+str(taryears[0])+' to '+str(taryears[1])+'...')
    tmpfiles = root1 + model[mm] + root2 + mrun[mm] + '/psl_interp' + '/tmp_interp_'+timestep+'rPlev_' + model[mm] +'_'+experiment+'_' + mrun[mm] + '_' + hemis +'*.nc' 
    nc = xr.open_mfdataset(tmpfiles)
    nc = nc.astype(precision)
    dates_all = nc.indexes['time']
    starthour_all = str(dates_all[0]).replace('-','').replace(' ','').replace(':','')
    endhour_all = str(dates_all[-1]).replace('-','').replace(' ','').replace(':','')
    
    #add applied interpolation technique as attribute
    nc.attrs['xesmf_regridding_method'] = regridding_method
    nc.attrs['calendar'] = 'standard'
    nc.attrs['source_calendar'] = calendar_from_nc
    nc.attrs['source_calendar_metadata'] = calendar_metadata
    
    #fill in global and variable attributes from source files (mainly from ESGF)
    for item in attrs_glob:
        nc.attrs[item[0]] = item[1]
    for item in attrs_psl:
        nc.psl.attrs[item[0]] = item[1]
    for item in attrs_time:
        nc.time.attrs[item[0]] = item[1]
    for item in attrs_lon:
        nc.lon.attrs[item[0]] = item[1]
    for item in attrs_lat:
        nc.lat.attrs[item[0]] = item[1]
        
    # savedir = root1 + model[mm] + root2 + mrun[mm] + '/psl_interp'
    # if os.path.isdir(savedir) != True:
        # os.makedirs(savedir)

    newfile = savedir + '/psl_interp_'+timestep+'rPlev_' + model[mm] +'_'+experiment+'_' + mrun[mm] + '_' + hemis +'.nc'
    nc.to_netcdf(newfile)
    nc.close()
    del(nc,getind,starthour_all,endhour_all,dates_all)
    #del(nc,getind,starthour_all,endhour_all)
    #delete the newly created temporary files
    print('INFO: deleting '+tmpfiles)
    for tt in tmpfiles_list:
        os.remove(tt)
    
    calendar_list.append(calendar_from_nc)
    print('INFO: Regridding for '+model[mm]+', '+experiment+', '+mrun[mm]+', '+hemis+', '+str(taryears).replace('[','').replace(']','').replace(', ',' to ')+' has been accomplished. See if any temporary files remain in '+tmpfiles)
    print(' ')
    print('------------ THE REGRIDDING PROCEDURE FOR '+model[mm]+', '+experiment+', '+mrun[mm]+', '+hemis+', '+str(taryears).replace('[','').replace(']','').replace(', ',' to ')+' WAS SUCCESSFUL -----------------------------------------')
    print(' ')
    
endtime = time.time()
elaptime = endtime - starttime
print('INFO: The following calendars were assinged; one for each GCM run:')
print(calendar_list)
print('interpolator_highres_py3.py has been run successfully! The elapsed time is '+str(elaptime)+'seconds, exiting now...')
quit()
