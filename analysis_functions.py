# -*- coding: utf-8 -*-

def get_location(location_f):
    '''get latitude and longitude (in decimals) for the location labelled by <location_f>, which can be any geograpical location i.e. not necessarily is a city.'''
    if location_f == 'Athens':
        tarlat_f = 37.983810
        tarlon_f = 23.727539
    elif location_f == 'Azores':
        tarlat_f = 38.305542
        tarlon_f = -30.384108
    elif location_f == 'Barcelona':
        tarlat_f = 41.390205
        tarlon_f = 2.154007
    elif location_f == 'Bergen':
        tarlat_f = 60.397076
        tarlon_f = 5.324383
    elif location_f == 'Cairo':
        tarlat_f = 30.033333
        tarlon_f = 31.233334
    elif location_f == 'Casablanca':
        tarlat_f = 33.589886
        tarlon_f = -7.603869
    elif location_f == 'Escorial': 
        tarlat_f = 	40.581608
        tarlon_f = -4.125124
    elif location_f == 'Madrid':
        tarlat_f = 40.4169644
        tarlon_f = -3.7035621
    elif location_f == 'NYC':
        tarlat_f = 40.71427
        tarlon_f = -74.00597
    elif location_f == 'Paris':
        tarlat_f = 48.85341
        tarlon_f = 2.3488
    elif location_f == 'Prague':
        tarlat_f = 50.0833
        tarlon_f = 14.4167
    elif location_f == 'Santander':
        tarlat_f = 43.462776
        tarlon_f = -3.805000
    elif location_f == 'SantiagoDC':
        tarlat_f = 42.8805
        tarlon_f = -8.5457
    elif location_f == 'Seattle':
        tarlat_f = 47.608013
        tarlon_f = -122.335167
    elif location_f == 'San_Francisco':
        tarlat_f = 37.773972
        tarlon_f = -122.431297
    elif location_f == 'Tokio':
        tarlat_f = 35.652832
        tarlon_f = 139.839478
    elif location_f == 'Weddell_Sea':
        tarlat_f = -69.51251
        tarlon_f = -54.75038
    elif location_f == 'Wellington':
        tarlat_f = -41.2889
        tarlon_f = 174.7772
    elif location_f == 'Zaragoza':
        tarlat_f = 	41.649693
        tarlon_f = -0.887712
    else:
        raise Exception('ERROR: check entry for <location_f>!')
    return(tarlat_f,tarlon_f)

def get_ensemble_config(ensemble_f,experiment_f):
    '''obtains ensemble configuration in terms of model labels (model_m) and run labels (mrun_m), both used for loading the netCDF data.
    The functions also returns a generalized ensemble label used for plotting and the target hours considered in the analyses, which may differ may differ from what is stored in netCDF format.'''
    if ensemble_f == 'cera20c' and experiment_f == '20c':
        model_f = ['cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c','cera20c']
        mrun_f = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']
        model_label_f = 'CERA-20C'
        tarhours_f = [0,3,6,9,12,15,18,21]
        # tarhours_f = [0,6,12,18] #note that the saved LWT data is 3h
    elif ensemble_f == 'era5' and experiment_f == '20c':
        model_f = ['era5','era5','era5','era5','era5','era5','era5','era5','era5','era5']
        mrun_f = ['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9']
        model_label_f = 'ERA5'
        tarhours_f = [0,3,6,9,12,15,18,21]
        # tarhours_f = [0,6,12,18] #note that the saved LWT data is 3h
    elif ensemble_f == 'mpi_esm_1_2_hr' and experiment_f == 'historical':
        model_f = ['mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr','mpi_esm_1_2_hr']
        mrun_f = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
        model_label_f = 'MPI-ESM1.2-HR'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'ec_earth3' and experiment_f == 'amip':
        model_f = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
        mrun_f = ['r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r9i1p1f1','r10i1p1f1']
        model_label_f = 'EC-Earth3'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'ec_earth3' and experiment_f == 'dcppA':
        model_f = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
        mrun_f = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
        # model_f = ['ec_earth3','ec_earth3']
        # mrun_f = ['r1i1p1f1','r2i1p1f1']
        model_label_f = 'EC-Earth3'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'ec_earth3' and experiment_f == 'historical':
        #model_f = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
        #mrun_f = ['r1i1p1f1','r3i1p1f1','r4i1p1f1','r7i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r20i1p1f1','r21i1p1f1','r23i1p1f1','r24i1p1f1','r25i1p1f1']
        model_f = ['ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3','ec_earth3']
        mrun_f = ['r1i1p1f1','r4i1p1f1','r10i1p1f1','r12i1p1f1','r14i1p1f1','r16i1p1f1','r17i1p1f1','r18i1p1f1','r19i1p1f1','r21i1p1f1'] #these are the runs which were concatenated with the ssp245 experiments to cover 1979-2028 using links
        # model_f = ['ec_earth3','ec_earth3']
        # mrun_f = ['r1i1p1f1','r4i1p1f1'] 
        model_label_f = 'EC-Earth3'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'ec_earth3_veg' and experiment_f == 'historical':
        model_f = ['ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg','ec_earth3_veg']
        mrun_f = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r10i1p1f1','r11i1p1f1']
        model_label_f = 'EC-Earth3-Veg'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'miroc6' and experiment_f == 'amip':
        model_f = ['miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6','miroc6']
        mrun_f = ['r1i1p1f1','r2i1p1f1','r3i1p1f1','r4i1p1f1','r5i1p1f1','r6i1p1f1','r7i1p1f1','r8i1p1f1','r9i1p1f1','r10i1p1f1']
        model_label_f = 'MIROC6'
        tarhours_f = [0,6,12,18]
    elif ensemble_f == 'mpi_esm_1_2_lr' and experiment_f == 'piControl':
        model_f = ['mpi_esm_1_2_lr','mpi_esm_1_2_lr']
        mrun_f = ['r1i1p1f1','r1i1p1f1']
        model_label_f = 'MPI-ESM1.2-LR'
        tarhours_f = [0,6,12,18]
    else:
        raise Exception('ERROR: unknown entry for <ensemble_f> and/or <experiment_f> input parameters !')
    return(model_f,mrun_f,model_label_f,tarhours_f)

def get_target_period(model_f,experiment_f,cmip_f=None,lead_time_f=None,lead_time_concept_f=None):
    '''returns the target period (taryears_f) and timestep (timestep_f) to be used in the analyses as a function of the gcm or reanalysis (<model_f>), experiment (<experiment_f>) and cmip generation (<cmip_f>)'''
    #define the time period the GCM data is interpolated for as a function of the experiment and considered GCM
    if experiment_f == 'amip' and cmip_f == 5:
        taryears_f=[1979,2005]
        timestep_f = '6h' #time step of the SLP data within the nc file from ESGF
    elif experiment_f == 'amip' and cmip_f == 6 and model_f not in ('ec_earth3'):
        taryears_f=[1979,2014]
        timestep_f = '6h'
    elif experiment_f == 'amip' and cmip_f == 6 and model_f == 'ec_earth3':
        taryears_f=[1979,2017]
        timestep_f = '6h'
    #for lead time concept FY
    elif experiment_f == 'dcppA' and lead_time_f == 1 and lead_time_concept_f == 'FY':
        taryears_f=[1961,2019]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 2 and lead_time_concept_f == 'FY':
        taryears_f=[1962,2020]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 3 and lead_time_concept_f == 'FY':
        taryears_f=[1963,2021]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 4 and lead_time_concept_f == 'FY':
        taryears_f=[1964,2022]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 5 and lead_time_concept_f == 'FY':
        taryears_f=[1965,2023]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 6 and lead_time_concept_f == 'FY':
        taryears_f=[1966,2024]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 7 and lead_time_concept_f == 'FY':
        taryears_f=[1967,2025]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 8 and lead_time_concept_f == 'FY':
        taryears_f=[1968,2026]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 9 and lead_time_concept_f == 'FY':
        taryears_f=[1969,2027]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 10 and lead_time_concept_f == 'FY':
        taryears_f=[1970,2028]
        timestep_f = '6h'
    #for lead time concept LT
    elif experiment_f == 'dcppA' and lead_time_f == 0 and lead_time_concept_f == 'LT':
        taryears_f=[1960,2019]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 1 and lead_time_concept_f == 'LT':
        taryears_f=[1961,2020]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 2 and lead_time_concept_f == 'LT':
        taryears_f=[1962,2021]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 3 and lead_time_concept_f == 'LT':
        taryears_f=[1963,2022]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 4 and lead_time_concept_f == 'LT':
        taryears_f=[1964,2023]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 5 and lead_time_concept_f == 'LT':
        taryears_f=[1965,2024]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 6 and lead_time_concept_f == 'LT':
        taryears_f=[1966,2025]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 7 and lead_time_concept_f == 'LT':
        taryears_f=[1967,2026]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 8 and lead_time_concept_f == 'LT':
        taryears_f=[1968,2027]
        timestep_f = '6h'
    elif experiment_f == 'dcppA' and lead_time_f == 9 and lead_time_concept_f == 'LT':
        taryears_f=[1969,2028]
        timestep_f = '6h'
    elif experiment_f == 'historical' and model_f == 'ec_earth3': #currently set to 1961-2028 for comparison of the historical+ssp245 runs with the dcppA runs
        print('WARNING: '+model_f+' '+experiment_f+' runs have been exceptionally extended with ssp245 runs to cover the 1961-2028 period! Turn this feature off in future versions of the <get_target_period()> function within analysis_functions.py!')
        #taryears_f = [1971,2028]
        taryears_f = [1961,2028]
        timestep_f = '6h'
    elif experiment_f == 'historical' and model_f in ('ec_earth3_veg','mpi_esm_1_2_hr'):
        taryears_f=[1850,2014]
        timestep_f = '6h'
    elif experiment_f == 'historical' and model_f not in ('ec_earth3_veg','mpi_esm_1_2_hr') and cmip_f == 6:
        taryears_f=[1979,2014]
        timestep_f = '6h'
    elif experiment_f == 'historical' and cmip_f == 5:
        taryears_f=[1979,2005]
        timestep_f = '6h'
    elif experiment_f == 'historical' and model_f in ('interim','jra55') and cmip_f == 1: #1 is here a fake number and means that reanalysis data is processsed instead of gcm data from cmip5 or 6
        taryears_f=[1979,2005]
        timestep_f = '6h'
    elif experiment_f == 'historical' and model_f == 'era5' and cmip_f == 1:
        taryears_f=[1940,2022]
        #timestep_f = '6h'
        timestep_f = '3h'
    elif experiment_f == 'piControl' and model_f == 'mpi_esm_1_2_lr':
        taryears_f=[1850,2261] #the max and min years are limited in pandas, see https://calmcode.io/til/pandas-timerange.html
        timestep_f = '6h'
    elif experiment_f in ('rcp85') and model_f == 'miroc_esm':
        taryears_f=[2006,2100]
        timestep_f = '6h'
    elif experiment_f in ('ssp126','ssp245','ssp370','ssp585'):
        taryears_f=[2015,2100]
        timestep_f = '6h'
    elif experiment_f == '20c' and model_f == 'cera20c':
        taryears_f=[1901,2010]
        timestep_f = '3h'
        #timestep_f = '6h'
    elif experiment_f == '20c' and model_f == 'era5':
        taryears_f=[1940,2022]
        timestep_f = '3h'
        #timestep_f = '6h'
    else:
        raise Exception('Error: check entry for <experiment_f>, <model_f> and/or <lead_time_f> !')
    return(taryears_f,timestep_f)

def z_transform(np_arr_f):
    '''performs standarization / z-transformation along the first dimension of an input numpy array (np_arr_f)'''
    if np_arr_f.ndim == 3:
        mn_arr_f = np.tile(np.mean(np_arr_f,axis=0),(np_arr_f.shape[0],1,1))
        std_arr_f = np.tile(np.std(np_arr_f,axis=0),(np_arr_f.shape[0],1,1))
    elif np_arr_f.ndim == 1:
        mn_arr_f = np.tile(np.mean(np_arr_f,axis=0),(np_arr_f.shape[0]))
        std_arr_f = np.tile(np.std(np_arr_f,axis=0),(np_arr_f.shape[0]))
    else:
        raise Exception('ERROR: z-transformation of an '+str(np_arr_f.ndim)+' dimensional numpy array is currently not supported !')
    z_arr_f = (np_arr_f - mn_arr_f) / std_arr_f
    return(z_arr_f)


def get_projection(region_loc,lats_loc,lons_loc):
    ''' obtain <mymap> variable needed for mapping the results obtained with analysis_hist.py and analysis_ensemble.py '''
    if region_loc in ('nh','nh_reduced'):
        mymap = Basemap(projection='npstere',boundinglat=np.min(lats_loc)-np.abs(np.diff(lats_values)[0])/2,lon_0=0)
    elif region_loc in ('eurocordex'):
        mymap = Basemap(projection='stere',width=7500000,height=5500000,lat_ts=np.mean(lats_loc),lat_0=np.mean(lats_loc),lon_0=np.mean(lons_loc))
    elif region_loc in ('escena'):
        mymap = Basemap(projection='stere',width=8000000,height=4500000,lat_ts=np.mean(lats_loc),lat_0=np.mean(lats_loc),lon_0=np.mean(lons_loc))
    elif region_loc == 'cordexna':
        mymap = Basemap(projection='stere',width=10000000,height=9000000,lat_ts=np.mean(lats_loc),lat_0=np.mean(lats_loc),lon_0=np.mean(lons_loc))
    elif region_loc == 'north_atlantic':
        mymap = Basemap(projection='stere',width=8500000,height=5600000,lat_ts=np.mean(lats_loc),lat_0=np.mean(lats_loc),lon_0=np.mean(lons_loc))
    elif region_loc == 'sh':
        #mymap = Basemap(projection='spstere',boundinglat=np.max(lats_loc),lon_0=0)
        mymap = Basemap(projection='spstere',boundinglat=np.max(lats_loc)+np.abs(np.diff(lats_values)[0])/2,lon_0=0)
    else:
        raise Exception('ERROR: check entry for <region_loc>')
    return(mymap)


def get_target_coords(region_loc):
    ''' obtain latlon coordinades needed to cut out the Lamb catalogues with analysis_hist.py and analysis_ensemble.py '''
    if region_loc == 'nh':
        latlims= [30.,90.] #latitude limits of the considered spatial domain (recall: Jones et al. 2013 suggest LWTs can be applied down to 30 degrees N)
        lonlims = [-180.,180.] #longitude limts of the domain
    elif region_loc == 'nh_reduced':
        latlims= [45.,90.]
        lonlims = [-180.,180.]
    elif region_loc == 'eurocordex':
        latlims = [27.,72.]
        lonlims = [-22.,50.]
    elif region_loc == 'medcordex': #from https://cordex.org/domains/region-12-mediterranean/
        latlims = [25.63,52.34]
        lonlims = [-20.21,50.85]
    elif region_loc == 'escena':
        latlims = [30.,60.]
        lonlims = [-40.,35.]
    elif region_loc == 'iberia':
        latlims = [35.,42.5]
        lonlims = [-10.,2.5]
    elif region_loc == 'cordexna':
        latlims = [28.,72.] #[12.,76.5]
        lonlims = [-172.,-22.]
    elif region_loc == 'north_atlantic': #From Perez et al. 2014, Clim. Dyn., doi: 10.1007/s00382-014-2078-8
        latlims = [30.,70.] # [25.,70.] in Perez et al. 2014 (nachgucken), [30.,70.] as alternative
        lonlims = [-70,15.] # [-70,15.] in Perez et al. 2014, [-70,15.] as alternative
    elif region_loc == 'sh':
        latlims = [-90.,-30.]
        lonlims = [-180.,180.]
    elif region_loc == 'sh_midlats': #southern hemisphere mid-latitudes
        latlims = [-60.,-50.]
        lonlims = [-180.,180.]
    else:
        raise Exception('ERROR: check entry for <region>!')
    return(latlims,lonlims)

def get_error_attrs(errortype):
    ''' get auxiliary variables need for plotting for a specific error type '''
    if errortype == 'MAE':
        cbounds_map = cbounds_mae
        errorunit = 'percent'
        lowerlim = 0.
        upperlim = 4. #upper limit for summary MAE boxplot's Y-axis; for hemispheric results, the best choice is 3.5 for MAE and 1.5 for KL, 3.5 for NH. For the special case of escena, jjas and timelag = 4, set upperlim = 4 to obtain a good representation in the boxplot
    elif errortype == 'KL':
        cbounds_map = cbounds_kl
        errorunit = 'entropy'
        lowerlim = 0.
        upperlim = 1.5
    elif errortype in ('TPMS'):
        cbounds_map = cbounds_tpms
        errorunit = 'probability'
        lowerlim = 0.
        upperlim = 0.15
    elif errortype in ('PERS'):
        cbounds_map = cbounds_pers
        errorunit = 'probability'
        lowerlim = 0
        upperlim = 1.4
    elif errortype in ('PERSall'):
        cbounds_map = cbounds_persall
        errorunit = 'probability'
        lowerlim = 0
        upperlim = 20 #15 for escena, jjas y timelag = 1; 20 for timelag = 4
    else:
        raise Exception('ERROR: check entry for <errortype>!')
    return(cbounds_map,errorunit,lowerlim,upperlim)
    

#def draw_error_map(plottype,region,lats_values_f,lons_values_f,mat_error,model_f,mrun_f,mylabels_f,colormap_f,halfres,cbounds_map,snapval,figpath,figfolder,errortype,refdata,correct_ru,alt_runs,figformat,errorunit,textsize,dpival,norm=None,ticks_cbar=None):
def draw_error_map(plottype,region,lats_values_f,lons_values_f,mat_error,colormap_f,halfres,cbounds_map,snapval,savename,titletext,cbarlabel,figformat,textsize,dpival,norm=None,ticks_cbar=None):
    ''' draws  and saves error or rank map for a given GCM'''
    fig = plt.figure()
    mymap = get_projection(region,lats_values_f,lons_values_f)
    ##add first latitude to 180 degrees
    #plotme, lons_plot = addcyclic(mat_error,lons_values)
    lons_values_f = lons_values_f-halfres #longitude must be corrected to depict cell centers, latitudes must not be corrected
    XX,YY = np.meshgrid(lons_values_f,lats_values_f)
    X, Y = mymap(XX, YY)
    plotme = np.ma.masked_where(np.isnan(mat_error),mat_error)
    if plottype == 'error':
        mymap.pcolormesh(X,Y, plotme, cmap=colormap_f, latlon=False, snap=snapval, rasterized=True) #in newer Python versions, the x and y coordinates are interpreted as cell centers.
        plt.title(titletext, size=textsize+1)
        cbar = plt.colorbar(shrink=0.75)
        cbar.set_label(cbarlabel, size=textsize)
        cbar.ax.tick_params(labelsize=textsize)
        #savepath=figpath+'/'+figfolder+'/'+region+'/maps/'+errortype+'_'+model_f+'_'+mrun_f+'_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.'+figformat
    elif plottype == 'rank':
        mymap.pcolormesh(X,Y, plotme, cmap=colormap_f, norm = norm, latlon=False, snap=snapval, rasterized=True)
        plt.title(titletext, size=textsize+1)
        cbar = plt.colorbar()
        cbar.set_label(cbarlabel, size=textsize)
        cbar.set_ticks(ticks_cbar)
        cbar.ax.tick_params(labelsize=textsize)
        #savepath=figpath+'/'+figfolder+'/'+region+'/maps/rank_'+model_f+'_'+mrun_f+'_wrt_'+refdata+'_'+region+'_ruout_'+correct_ru+'_altruns_'+alt_runs+'_1979-2005.'+figformat

    mymap.drawcoastlines(color='black')
    plt.savefig(savename, dpi=dpival)
    plt.close('all')
    

def get_transition_probabilities(wtcount,wt,timelag):
    #transition probabilities for reanalysis data###########################    
    #move array containing observed LWTs by timelag along the time axis and add nan rows
    wt_lag = wt[timelag:,:,:]
    nan_rows = np.zeros((timelag,wt.shape[1],wt.shape[2]))
    wt_lag = np.concatenate((wt_lag,nan_rows),axis=0)
    #initiate the 4d array containing the transition probablities and fill it iteratively.
    prob_4d = np.zeros((wt.shape[1],wt.shape[2],len(wtcount),len(wtcount)))
    for wwi in list(range(len(wtcount))):
        for wwj in list(range(len(wtcount))):
            wt_step = np.float32(np.copy(wt))
            wt_lag_step = np.float32(np.copy(wt_lag))
            nanind = wt_step != wtcount[wwi]
            nanind_lag = wt_lag_step != wtcount[wwj]
            wt_step[nanind] = np.nan
            wt_lag_step[nanind_lag] = np.nan
            wt_sum = wt_step + wt_lag_step
            wt_sum[wt_sum >= 1.] = 1.
            prob_lonlat = np.nansum(wt_sum,axis=0)/wt_sum.shape[0]
            prob_4d[:,:,wwi,wwj] = prob_lonlat
            #print('Transition probabilities for LWT combination '+str(wtcount[wwi])+' and '+str(wtcount[wwj])+' completed!')
    return(prob_4d)


def get_fig_transprobmat(prob_4d,lons_values,lats_values,cmap_probmat,edgecolors,figformat,modellabel,figpath_f,region,dpival,timelag,wt_names):
    wtcount = np.arange(1,len(wt_names)+1) #pcolor needs one dimension more than in the data to work well, see https://github.com/matplotlib/basemap/issues/107
    for ii in range(len(lons_values)):
        for jj in range(len(lats_values)):
            XX,YY = np.meshgrid(wtcount,wtcount)
            #fig = plt.pcolor(np.array(wtcount)-0.5,np.array(wtcount)-0.5,prob_4d[ii,jj,:,:],cmap=cmap_probmat,vmin=0,vmax=np.percentile(prob_4d[ii,jj,:,:].flatten(),99.5),edgecolors=edgecolors)
            fig = plt.pcolor(XX,YY,prob_4d[ii,jj,:,:],cmap=cmap_probmat,vmin=0,vmax=np.percentile(prob_4d[ii,jj,:,:].flatten(),99.5),edgecolors=edgecolors)
            plt.axis('square')
            plt.colorbar()
            plt.xlabel('Transition from')
            plt.xticks(np.array(wtcount))
            #plt.xticks(np.array(wtcount)+0.5)
            fig.axes.set_xticklabels(np.array(wt_names),rotation=45.,size=8.)
            plt.ylabel('Transition to')
            #plt.yticks(np.array(wtcount)+0.5)
            plt.yticks(np.array(wtcount))
            fig.axes.set_yticklabels(np.array(wt_names),rotation=0.,size=8.)
            plt.title('Transition probability for '+modellabel+' at lon '+str(lons_values[ii])+', lat '+str(lats_values[jj])+', lag '+str(timelag*6)+'h')
            savepath = figpath_f+'/trans_prob_'+modellabel+'_'+region+'_lon_'+str(lons_values[ii])+'_lat_'+str(lats_values[jj])+'_lag_'+str(timelag*6)+'h.'+figformat
            plt.savefig(savepath, dpi = dpival)
            plt.close('all')

def lin_detrend(xr_ar,rm_mean_f):
    """also used in pySeasonal package; performs linear detrending of the xarray DataArray xr_ar along the time dimension, rm_mean_f specifies whether the mean is removed yes or no"""
    coeff = xr_ar.polyfit(dim='time',deg=1) #deg = 1 for linear detrending
    fit = xr.polyval(xr_ar['time'], coeff.polyfit_coefficients)
    if rm_mean_f == 'yes':
        xr_ar_detrended = xr_ar - fit
    elif rm_mean_f == 'no':
        tiles_f = np.ones(len(xr_ar.dims))
        tiles_f[0] = len(xr_ar.time)
        meanvals_f = np.tile(xr_ar.mean(dim='time'),tiles_f.astype('int'))
        xr_ar_detrended = xr_ar - fit + meanvals_f
    else:
        raise Exception('ERROR: check entry for <rm_mean_f> input parameter!')
    return(xr_ar_detrended)

def load_and_aggregate_wts(filename,region_f,taryears,tarwts,aggregation,timestep,latweighting,tarmonths_f):
    #load and process wt data
    wt_f = xr.open_dataset(filename)
    
    #filter the target region
    latlims,lonlims = get_target_coords(region_f)
    print('INFO: filtering out the target region '+region_f+'. Latitude limits are '+str(latlims)+', longitude limits are '+str(lonlims)+'.')
    latind = np.where((wt_f.lat >= latlims[0]) & (wt_f.lat <= latlims[1]))[0]
    lonind = np.where((wt_f.lon >= lonlims[0]) & (wt_f.lon <= lonlims[1]))[0]
    wt_f = wt_f.isel(lat = latind, lon = lonind)
    lats_f = wt_f.lat.values
    lons_f = wt_f.lon.values
 
    #select target years and months
    dates_f = pd.DatetimeIndex(wt_f.time.values)
    #year_ind_wt = np.where((dates_f.year >= taryears[0]) & (dates_f.year <= taryears[1]))[0]
    year_ind_wt = np.where((dates_f.year >= taryears[0]) & (dates_f.year <= taryears[1]) & dates_f.month.isin(tarmonths_f))[0]
    wt_f = wt_f.isel(time=year_ind_wt)
    dates_f = dates_f[year_ind_wt]
    
    bin_array_f = np.zeros(wt_f.wtseries.shape)
    print('INFO: Searching the requested LWTs '+str(tarwts))
    start = time.time()
    #tarwt_ind = np.where(wt_f.wtseries.isin(tarwts))
    tarwt_ind = wt_f.wtseries.isin(tarwts)
    bin_array_f[tarwt_ind] = 1
    end = time.time()
    print('INFO: The search was accomplished within '+str(end-start)+' seconds.')
    bin_array_f = bin_array_f.reshape(bin_array_f.shape[0],bin_array_f.shape[1]*bin_array_f.shape[2])
    arr_tarwts_f = xr.DataArray(data=bin_array_f,coords =[dates_f,np.arange(len(wt_f.lon.values)*len(wt_f.lat.values))],dims=['time','gridboxes'],name='wtseries')
    
    #get time series with yearly target WT counts
    if aggregation == 'year':
        arr_tarwts_f = arr_tarwts_f.groupby('time.year').sum('time') #calculate annual mean values
        dates_f = arr_tarwts_f.year
    elif aggregation in ('1','5','7','10','30'): #aggregation in days
        if timestep == '3h':
            interval_f = int(float(aggregation)*8)
        elif timestep == '6h':
            interval_f = int(float(aggregation)*4)
        else:
            raise Exception('Error: unknown entry for <timestep> !')
        arr_tarwts_f = arr_tarwts_f.rolling(time=interval_f,center=True).sum().dropna('time') #calculate rolling daily mean values
        arr_tarwts_f = arr_tarwts_f[0::interval_f] #remove overlapping data
        dates_f = pd.DatetimeIndex(arr_tarwts_f.time)
        print('INFO: the lenght of the time series is '+str(arr_tarwts_f.shape[0]))
    else:
        raise Excpetion('ERROR: unknown entry for <aggregation>!')
    
    #get latitude weights
    xx,yy = np.meshgrid(lons_f,lats_f)
    yy = yy.reshape((len(lons_f)*len(lats_f)))
    yy = np.tile(yy,(arr_tarwts_f.shape[0],1))
    
    #get areal mean values optionally weighting by latitude
    if latweighting == 'yes':
        print('INFO: As requested by the user, the weighted areal mean is calculated. The point-wise WT counts are weighted by the cosine of latitude to take into account the convergence of the meridians.')
        weights = np.cos(np.radians(yy))
        arr_tarwts_f = arr_tarwts_f*weights    
        arr_tarwts_f = arr_tarwts_f.sum(axis=1)/np.sum(weights,axis=1)
    else:
        arr_tarwts_f = arr_tarwts_f.mean(axis=1)
    
    wt_f.close()    
    del(wt_f,bin_array_f,year_ind_wt)
    #return time aggregations for areal mean WT counts and the corresponding pandas DatetimeIndex
    return(arr_tarwts_f,dates_f)

def load_and_aggregate_wts_anom(filename,region_f,taryears,tarwts,aggregation,timestep,latweighting,tarmonths_f,rollw_f):
    #load and process wt data
    wt_f = xr.open_dataset(filename)
    
    #filter the target region
    latlims,lonlims = get_target_coords(region_f)
    print('INFO: filtering out the target region '+region_f+'. Latitude limits are '+str(latlims)+', longitude limits are '+str(lonlims)+'.')
    latind = np.where((wt_f.lat >= latlims[0]) & (wt_f.lat <= latlims[1]))[0]
    lonind = np.where((wt_f.lon >= lonlims[0]) & (wt_f.lon <= lonlims[1]))[0]
    wt_f = wt_f.isel(lat = latind, lon = lonind)
    lats_f = wt_f.lat.values
    lons_f = wt_f.lon.values

    #select target years and months
    dates_f = pd.DatetimeIndex(wt_f.time.values)
    #year_ind_wt = np.where((dates_f.year >= taryears[0]) & (dates_f.year <= taryears[1]))[0]
    year_ind_wt = np.where((dates_f.year >= taryears[0]) & (dates_f.year <= taryears[1]) & dates_f.month.isin(tarmonths_f))[0]
    wt_f = wt_f.isel(time=year_ind_wt)
    dates_f = dates_f[year_ind_wt]
    
    bin_array_f = np.zeros(wt_f.wtseries.shape)
    print('INFO: Searching the requested LWTs '+str(tarwts))
    start = time.time()
    #tarwt_ind = np.where(wt_f.wtseries.isin(tarwts))
    tarwt_ind = wt_f.wtseries.isin(tarwts)
    bin_array_f[tarwt_ind] = 1
    end = time.time()
    print('INFO: The search was accomplished within '+str(end-start)+' seconds.')
    bin_array_f = bin_array_f.reshape(bin_array_f.shape[0],bin_array_f.shape[1]*bin_array_f.shape[2])
    arr_tarwts_f = xr.DataArray(data=bin_array_f,coords =[dates_f,np.arange(len(wt_f.lon.values)*len(wt_f.lat.values))],dims=['time','gridboxes'],name='wtseries')
    
    #get time series with yearly target WT counts
    if aggregation == 'year':
        arr_tarwts_f = arr_tarwts_f.groupby('time.year').sum('time') #calculate annual mean values
        dates_f = arr_tarwts_f.year
    elif aggregation in ('1','5','7','10','30'): #aggregation in days
        if timestep == '3h':
            interval_a = 8 #interval used for anomaly calculation, number of timesteps per day used to get daily counts, which are then used to calc. anomalies from daily climatologcial mean counts
        elif timestep == '6h':
            interval_a = 4 #interval used for anomaly calculation, number of timesteps per day used to get daily counts, which are then used to calc. anomalies from daily climatologcial mean counts
        else:
            raise Exception('Error: unknown entry for <timestep> !')
        interval_f = int(float(aggregation)*1) #here used to aggregate the daily areal mean counts obtained hereafter
        
        #daily counts used to derived anomalies
        #daycount = arr_tarwts_f.rolling(time=interval_a,center=True).sum().dropna('time') #absolute
        daycount = arr_tarwts_f.rolling(time=interval_a,center=True).sum().dropna('time')/interval_a*100 #percentage of the day covered with this LWT 
        daycount = daycount[0::interval_a] #remove overlapping data
        dates_a = pd.DatetimeIndex(daycount.time)
        
        #get latitude weights
        xx,yy = np.meshgrid(lons_f,lats_f) #get lon and lat matrices
        yy = yy.reshape((len(lons_f)*len(lats_f))) #transform to vector
        yy_a = np.tile(yy,(daycount.shape[0],1)) #get matrix with n time steps and m grid-boxes from the vector above
        weights_a = np.cos(np.radians(yy_a))
        
        #get areal mean wt count per day
        if latweighting == 'yes':
            print('INFO: As requested by the user, the weighted areal mean is calculated. The point-wise WT counts are weighted by the cosine of latitude to take into account the convergence of the meridians.')
            daycount = daycount*weights_a
            daycount = daycount.sum(axis=1)/np.sum(weights_a,axis=1)
        else:
            print('INFO: As requested by the user, non-weighted areal average WT counts are calculated')
            daycount = daycount.mean(axis=1)
        daycount_clim = daycount.groupby('time.dayofyear').mean('time') #calculate climatlogical daily mean values
        daycount_clim_aux = xr.concat((daycount_clim,daycount_clim),dim='dayofyear') #concat 2x366 values to avoid nan while calculating the running mean values hereafter
        
        ##optionally smooth the daily climatological means with a rolling mean
        tar_intv = np.floor(rollw_f/2)
        tar_i = np.linspace(tar_intv*-1,tar_intv-1,rollw_f-1).astype(int)
        sel_i = np.linspace(daycount_clim.shape[0]-tar_intv,daycount_clim.shape[0]+tar_intv-1,rollw_f-1).astype(int)
        daycount_clim = daycount_clim.rolling(dayofyear=rollw_f,center=True).mean()
        daycount_clim_aux = daycount_clim_aux.rolling(dayofyear=rollw_f,center=True).mean()
        daycount_clim[tar_i] = daycount_clim_aux[sel_i] 
        
        #daycount_clim[0] = daycount_clim_aux.shape[0]
        #daycount_clim[1] = daycount_clim_aux.shape[0]+1
        #daycount_clim[-1] = daycount_clim_aux.shape[0]-1
        #daycount_clim[-2] = daycount_clim_aux.shape[0]-2
        
        #missval = (daycount_clim[-3]+daycount_clim[2])/2
        #daycount_clim[[0,1]] = missval
        #daycount_clim[[-2,-1]] = missval        

        #cacluated anomaly values on basis of daily climatological mean counts 
        daycount_a = daycount.copy()
        for dy in np.arange(len(daycount_clim.dayofyear)):
            dayind = np.array(dates_a.dayofyear)==daycount_clim.dayofyear[dy].values
            daycount_a[dayind] = daycount[dayind] - daycount_clim[dy]
            del(dayind)
    
        #counts over the temporal interval specified by the user in <aggregation>
        daycount_a = daycount_a.rolling(time=interval_f,center=True).sum().dropna('time') #calculate rolling daily mean values
        daycount_a = daycount_a[0::interval_f] #remove overlapping data
        dates_a = pd.DatetimeIndex(daycount_a.time)
        print('INFO: the lenght of the time series is '+str(daycount_a.shape[0]))
    else:
        raise Excpetion('ERROR: unknown entry for <aggregation>!')
        
    #get output variable name consistent to <load_and_aggregate_wts()>
    arr_tarwts_a = daycount_a.copy()
    wt_f.close()    
    del(wt_f,bin_array_f,year_ind_wt,daycount_clim_aux)
    #return time aggregations for areal mean WT counts and the corresponding pandas DatetimeIndex
    return(arr_tarwts_a,dates_a,daycount_clim)

#def plot_power_spectrum(runmode_f,f_f,Pxx_f,Pxx_ci,samplefreq,window,nfft_f,detrend,anom,wtnames,tarwts,modellabel,runlabel,taryears,seaslabel,aggreg_f,region,outformat,dpival,titlesize): #former version
def plot_power_spectrum(runmode_f,f_f,Pxx_f,Pxx_ci,samplefreq,window,nfft_f,detrend,aggreg_f,savename_f,titlelabel_f,dpival,titlesize):
    #plots power and cross power spectra depending on <runmode_f>
    wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")
    if runmode_f == 'single':
        #titlelabel_f = 'PS '+wtlabel+' '+modellabel.upper()+' '+str(runlabel)+'m '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg_f+' nfft'+str(nfft_f)+' dtrend'+detrend+' anom'+anom+' '+region
        ylabel_f = 'Power Spectrum Amplitude'
        #savename_f = figs+'/'+aggreg_f+'/periodogram/'+modellabel+'_'+str(runlabel)+'_members_periodogram_LWT_'+wtlabel.replace(" ","_")+'_'+str(runlabel)+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg_f+'_nfft'+str(nfft_f)+'_dtrend_'+detrend+'_anom_'+anom+'_'+region+'.'+outformat
    elif runmode_f == 'cross':
        #titlelabel_f = 'CPS '+wtlabel+' '+modellabel.upper()+' '+str(runlabel)+'m '+str(taryears[0])+'-'+str(taryears[1])+' '+seaslabel+' '+aggreg_f+' nfft'+str(nfft_f)+' dtrend'+detrend+' anom'+anom+' '+region1+region2
        ylabel_f = 'Cross Power Spectrum Amplitude'
        #savename_f = figs+'/'+aggreg_f+'/periodogram/'+modellabel+'_'+str(runlabel)+'_members_cross_spectrum_LWT_'+wtlabel.replace(" ","_")+'_'+str(runlabel)+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+seaslabel+'_'+aggreg_f+'_nfft'+str(nfft_f)+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+'_'+region2+'.'+outformat
    else:
        raise Exception('ERROR: check entry for <runmode>!')

    period_yr = 1/f_f
    #period_yr[period_yr == np.inf] = np.nan #set inf to nan
    maxamp_ind = np.argsort(np.abs(Pxx_f))
    outind = period_yr[maxamp_ind] == np.inf #find inf values
    maxamp_ind = np.delete(maxamp_ind,outind) #and remove them
    fig = plt.figure()
    plt.plot(f_f, Pxx_f,linewidth=1)
    #plot confidence interval if available
    if len(Pxx_ci) > 0: #check whether Pxx_ci is empty ([])
        #f_ci = np.tile(f_f,(Pxx_ci.shape[0],1))
        plt.plot(f_f, Pxx_ci[0,:],color='grey',linewidth=1,linestyle='dotted')
        plt.plot(f_f, Pxx_ci[1,:],color='grey',linewidth=1,linestyle='dashed')
        plt.plot(f_f, Pxx_ci[2,:],color='grey',linewidth=1,linestyle='dashed')
        plt.plot(f_f, Pxx_ci[3,:],color='grey',linewidth=1,linestyle='dotted')
    #plot values as text for the 10 largest amplitudes in the spectrum
    for ii in maxamp_ind[-10:]:
        if aggreg_f == 'year':
            textme = str(np.round(period_yr[ii],2))
        else:
            textme = str(int(np.round(period_yr[ii])))
        plt.text(f_f[ii],Pxx_f[ii],textme)
    
    ##plot y-axis in log scale
    #plt.yscale('log')
    if aggreg_f == 'year':
        ag_label = 'years'
        x_labels = np.round(period_yr,2)
        rotation = 90.
    else:
        ag_label = aggreg_f+' days'
        x_labels = np.round(period_yr)
        rotation = 0.
        
    plt.xlabel('Period ('+ag_label+')')
    plt.xticks(ticks=f_f,labels=x_labels,rotation=rotation,fontsize=6)
    plt.ylabel(ylabel_f)
    plt.title(titlelabel_f,size=titlesize)
    plt.savefig(savename_f,dpi=dpival)
    plt.close('all')
    print('The 10 periods (in years) with largest absolute amplitude in ascending order are:')
    print(period_yr[maxamp_ind[-10:]])

def plot_seas_cycle(seas_cyc_f,detrend,anom,wtnames,tarwts,modellabel,runlabel,taryears,aggreg_f,region,outformat,dpival,titlesize):
    '''plots monthly climatological mean WT frequencies'''
    fig = plt.figure()
    plt.bar(seas_cyc_f.month.values,seas_cyc_f.values)
    plt.ylim(seas_cyc_f.values.min()-0.2,seas_cyc_f.values.max()+0.2)
    wtlabel = str(np.array(wtnames)[np.array(tarwts)-1]).replace("[","").replace("]","").replace("'","")
    plt.title('LWT '+wtlabel+' '+modellabel.upper()+' '+str(runlabel)+' members '+str(taryears[0])+'-'+str(taryears[1])+' '+aggreg_f+' dtrend '+detrend+' anom '+anom+' '+region1,size=titlesize)
    savename = figs+'/'+aggreg_f+'/seascyc/'+modellabel+'_'+str(runlabel)+'_seasonal_cycle_LWT_'+wtlabel.replace(" ","_")+'_'+str(taryears[0])+'_'+str(taryears[1])+'_'+aggreg_f+'_dtrend_'+detrend+'_anom_'+anom+'_'+region1+'.'+outformat
    plt.savefig(savename,dpi=dpival)
    plt.close('all')

#def plot_time_series(time_series_f,detrend,anom,wtnames,tarwts,modellabel,runlabel,taryears,seaslabel,aggreg_f,regionlabel,outformat,dpival,titlesize,cutoff,meanperiod): #old format
def plot_time_series(time_series_f,detrend,aggreg_f,savename_f,titlelabel_f,dpival,cutoff=None,titlesize=None):
    """plots time series of i models / reanalyses (or runs thereof) and j time-steps"""
    print('you are here')
    if aggreg_f == 'year':
        time_axis = time_series_f.time.values
        time_labels = time_axis
        time_title = 'year'
    elif aggreg_f in ('1','5','7','10','30'):
        time_axis = np.arange(len(time_series_f.time.values))
        time_labels = np.array([time_series_f.time[ii].values.astype(str)[0:10] for ii in np.arange(len(time_series_f.time))])
        time_title = aggreg_f+'-day periods since '+str(time_series_f.time.values[0])[0:10]
    else:
        raise Exception('ERROR: Unknown entry for <aggreg_f>')
    #only plot unitl number of timesteps defined in <cutoff>
    if np.isscalar(cutoff):
        time_axis = time_axis[0:cutoff]
        time_labels = time_labels[0:cutoff]
        time_series_f = time_series_f[:,0:cutoff]
        time_labels[1::2] = ''
    else:
        print('INFO: The <cutoff> input parameter was set to '+str(cutoff)+' and is thus not applied.')
        
    means =  time_series_f.mean(axis=0)
    runmeans = means.rolling(time=meanperiod,center=True).mean()
    
    time_labels = time_labels.tolist()
    fig = plt.figure()
    plt.plot(np.transpose(np.tile(time_axis,(time_series_f.values.shape[0],1))),np.transpose(time_series_f.values),linewidth=0.5)
    #plt.plot(time_axis[len(time_axis)-len(runmeans):],runmeans,linewidth=0.5,color='red')
    plt.plot(time_axis,runmeans,linewidth=0.5,color='red')
    plt.xticks(ticks=time_axis,labels=time_labels,rotation=90.,fontsize=4.)
    plt.xlim(time_axis[0],time_axis[-1])
    plt.ylabel(timestep+' areal mean occurrence frequency')
    plt.ylim(time_series_f.min(),time_series_f.max())
    plt.title(titlelabel_f,size=titlesize)
    plt.savefig(savename_f,dpi=dpival)
    plt.close('all')


def get_lowfreq_var(time_series_f,mktest_f,testlevel_f,periodogram_type_f,nfft_f,fs_f,window_f,scaling_f,detrend_f,repetitions_f,ci_percentile_f,nperseg = None):
    '''plots maps showing 1) point-wise long-term tendencies and 2) power spectra for different periods, as well as their significance and ensemble agreement on slope (for trends only) and significance (for both trends and power spectra).'''
    #print('nperseg is '+str(nperseg))
    if mktest == 'original':
        trend = mk.original_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'yue_wang':
        trend = mk.yue_wang_modification_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'hamed_rao':
        trend = mk.hamed_rao_modification_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'pre_white':
        trend = mk.pre_whitening_modification_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'regional':
        trend = mk.regional_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'seasonal':
        trend = mk.seasonal_test(time_series_f,alpha=testlevel_f)
    elif mktest == 'theilslopes':
        trend = stats.theilslopes(time_series_f,alpha=testlevel_f)
        trend.h = np.sum(np.sign((trend.low_slope,trend.high_slope))) != 0
        trend.p = np.nan
    else:
        raise Exception('ERROR: unknown entry for <mktest>!')
                        
    #calc. Power spectra and test them for significance
    ntime = len(time_series_f) #recall that means is an xr DataArray
    #get power spectrum
    if periodogram_type == 'periodogram':
        f_out, Pxx_out = signal.periodogram(time_series_f, fs=fs_f, nfft=nfft_f, window=window_f, scaling=scaling_f, detrend=detrend_f) #transforms to anomalies prior to calculating frequencies and spectrum by default
        #get confidence intervals from randomly reshuffled time series
        #init array to be filled with power spectra from randomly reshuffled time series
        Pxx_rand = np.zeros((repetitions_f,len(Pxx_out)))
        #print('INFO: Init power spectrum calculations with randomly reshuffled times series and '+str(repetitions)+' repetitions...')
        for rr in list(np.arange(repetitions_f)):
            rand1 = np.random.randint(0,ntime,ntime)
            f_rand_out, Pxx_rand_out = signal.periodogram(time_series_f[rand1], fs=fs_f, nfft=nfft_f, window=window_f, scaling=scaling_f, detrend=detrend_f)
            Pxx_rand[rr,:] = Pxx_rand_out
        ##get 95% confidence intervals for the random power spectra
    elif periodogram_type == 'Welch': #see https://het.as.utexas.edu/HET/Software/Scipy/generated/scipy.signal.welch.html
        f_out, Pxx_out = signal.welch(time_series_f, fs=fs_f, nfft=nfft_f, nperseg=nperseg, noverlap=None, window=window_f, scaling=scaling_f, detrend=detrend_f) #transforms to anomalies prior to calculating frequencies and spectrum by default
        #get confidence intervals from randomly reshuffled time series
        #init array to be filled with power spectra from randomly reshuffled time series
        Pxx_rand = np.zeros((repetitions_f,len(Pxx_out)))
        #print('INFO: Init power spectrum calculations with randomly reshuffled times series and '+str(repetitions)+' repetitions...')
        for rr in list(np.arange(repetitions_f)):
            rand1 = np.random.randint(0,ntime,ntime)
            f_rand_out, Pxx_rand_out = signal.welch(time_series_f[rand1], fs=fs_f, nfft=nfft_f, nperseg=nperseg, noverlap=None, window=window_f, scaling=scaling_f, detrend=detrend_f)
            Pxx_rand[rr,:] = Pxx_rand_out
    else:
        raise Exception('ERROR: unknown value for <periodgram_type>!')
    
    ##get upper limit of the confidence interval for the random power spectra
    ci_out = np.percentile(Pxx_rand,ci_percentile_f,axis=0)
    return(trend,f_out,Pxx_out,ci_out)

def get_csd_var(arr1_f,arr2_f,ntime_f,fs_f,window_f,detrend_f):
    '''Calculates random resamples from numpy arrays arr1_f and arr2_f (both 3d: time x lat x lon or similar) that can be used for critical value estimation of the cross power spectra CPXX_rand_f'''
    rand1 = np.random.randint(0,ntime_f,ntime_f)
    fq_rand_f, CPxx_rand_f = signal.csd(arr1_f[rand1,:,:], arr2_f[rand1,:,:], fs=fs_f, window=window_f, nperseg=arr1_f.shape[0], noverlap=None, detrend=detrend_f, return_onesided=True, scaling='spectrum', axis=0, average='mean')
    return(CPxx_rand_f)

def get_csd_ar(arr1d_f,arr3d_f,nangrid_f,ntime_f,acorr_lag_f,fs_f,window_f,nfft_f,detrend_f):
    '''Calculates critical values for a cross-spectrum analysis between a standardized time series reprsented by the statsmodel object AR_obj1_f and an 3d AR(acorr_lag_f) process array derived from arr3d_f (which is a 3d numpy array containing standardized time series)''' 
    np.seterr(divide='ignore', invalid='ignore') #ignore devide by zero warnings otherwise returned by each sub-worker (cpu-core) if this function is called in parallel mode
    
    acorr_object1 = sm.tsa.acf(arr1d_f, nlags = ntime_f)
    par1_object1 = np.array([1, acorr_object1[acorr_lag_f]*-1])
    par2_object1 = np.array([1])
    AR_object1 = ArmaProcess(par1_object1, par2_object1)
    rand_arr1 = AR_object1.generate_sample(nsample=ntime_f)
    #rand_arr1 = get_random_sample(AR_object1,ntime_f)
    rand_arr1 = np.tile(np.expand_dims(np.expand_dims(rand_arr1,axis=-1),axis=-1),[1,arr3d_f.shape[1],arr3d_f.shape[2]])
            
    rand_arr2 = np.zeros(arr3d_f.shape)
    for ii in np.arange(arr3d_f.shape[1]):
        for jj in np.arange(arr3d_f.shape[2]):
            if nangrid[ii,jj] == True:
                #print('Info: nans are present at '+str(ii)+' and '+str(jj)+'. Skipping...')
                continue
            acorr_object2 = sm.tsa.acf(arr3d_f[:,ii,jj], nlags = ntime_f)
            par1_object2 = np.array([1, acorr_object2[acorr_lag_f]*-1]) #ArmaProcess() function called below requires to switch the sign of the autocorrelation coefficient 
            par2_object2 = np.array([1])
            AR_object2 = ArmaProcess(par1_object2, par2_object2)
            rand_arr2_step = AR_object2.generate_sample(nsample=ntime_f)
            #rand_arr2_step = get_random_sample(AR_object2,ntime_f)
            if np.any(np.isnan(rand_arr2_step)): #rand_arr2_step may contain nans that cannot be processed by signal.csd(), so skip in nan is found a one or mor timesteps
                continue
            rand_arr2[:,ii,jj] = rand_arr2_step

    fq_rand_f, CPxx_rand_f = signal.csd(rand_arr1, rand_arr2, fs=fs_f, window=window_f, nperseg=nfft_f, noverlap=None, detrend=detrend_f, return_onesided=True, scaling='spectrum', axis=0, average='mean')
    return(CPxx_rand_f)
    del CPxx_rand_f

def get_random_sample(statsmodel_object,ntime_f):
    '''generates random sample from an object describing an ArmaProcess obtained with the get_csd_ar() function defined above.'''
    #rng_qrng.check_random_state(seed=None)
    random.seed(None)
    rand_sample_f = statsmodel_object.generate_sample(nsample=ntime_f)
    return(rand_sample_f)

#def get_map_lowfreq_var(pattern_f,xx_f,yy_f,agree_ind_f,minval_f,maxval_f,dpival_f,title_f,savename_f,halfres_f,colormap_f,titlesize_f,units_f): #former version
#def get_map_lowfreq_var(pattern_f,xx_f,yy_f,agree_ind_f,minval_f,maxval_f,dpival_f,title_f,savename_f,halfres_f,colormap_f,titlesize_f,cbarlabel_f,map_proj_f,origpoint=None):
def get_map_lowfreq_var(pattern_f,xx_f,yy_f,minval_f,maxval_f,dpival_f,title_f,savename_f,halfres_f,colormap_f,titlesize_f,cbarlabel_f,map_proj_f,agree_ind=None,origpoint=None,cbar_norm_f=None):
    '''origpoint refers to a single point to be plotted on the map, e.g. the single point where LWT counts are associated with SST grid-boxes around the World'''
    fig = plt.figure()
    
    # #this next code passage is likely not used by any script any more and can probably be removed in the future
    # maxind = np.argsort(pattern_f.flatten())[-1]
    # max_x = xx_f.flatten()[maxind]
    # max_y = yy_f.flatten()[maxind]
    # minind = np.argsort(pattern_f.flatten())[0]
    # min_x = xx_f.flatten()[minind]
    # min_y = yy_f.flatten()[minind]

    if cbar_norm_f == None:
        print('No norm is applied to the colorbar.')
        norm_f = None
    elif cbar_norm_f > 0:
        boundaries_f = np.linspace(-0.5, cbar_norm_f+0.5, cbar_norm_f+2)
        cmap_discrete_f = plt.get_cmap('hot_r')
        #cmap_discrete_f = plt.get_cmap('rainbow')
        #cmap_discrete_f = plt.get_cmap('jet')
        norm_discrete_f = mcolors.BoundaryNorm(boundaries_f, cmap_discrete_f.N)
    else:
        raise ValueError("Unknown entry for <cbar_norm_f> !")

    ax = fig.add_subplot(111, projection=map_proj_f)
    ax.set_extent([xx_f.min()-halfres, xx_f.max()+halfres_f, yy_f.min()-halfres_f, yy_f.max()+halfres_f], ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE, zorder=4, color='black')
    
    if cbar_norm_f == None:
        image = ax.pcolormesh(xx_f, yy_f, pattern_f, vmin=minval_f, vmax=maxval_f, cmap=colormap_f, transform=ccrs.PlateCarree(), shading = 'nearest', zorder=3, rasterized=True)
    elif cbar_norm_f > 0:
        image = ax.pcolormesh(xx_f, yy_f, pattern_f, cmap=cmap_discrete_f, transform=ccrs.PlateCarree(), shading = 'nearest', zorder=3, norm=norm_discrete_f, rasterized=True)
    else:
        raise ValueError("Unknown entry for <cbar_norm_f> !")        
        
    #image = ax.contourf(xx_f, yy_f, pattern_f, vmin=minval_f, vmax=maxval_f, cmap=colormap_f, transform=ccrs.PlateCarree(), shading = 'nearest', zorder=3)
    
    #get size of the points indicating significance
    if halfres_f < 1.:
        pointsize_f = 0.25 #0.5
        marker_f = '+'
    else:
        pointsize_f = 1 #0.25
        marker_f = '.' #o
    
    if agree_ind is None:
        print('No <agree_ind> was found by get_map_lowfreq_var() function !')
    else:
        toplayer_x = xx_f.flatten()[agree_ind.flatten()]
        toplayer_y = yy_f.flatten()[agree_ind.flatten()]
        ax.plot(toplayer_x, toplayer_y, color='black', marker=marker_f, linestyle='none', markersize=pointsize_f, transform=ccrs.PlateCarree(), zorder=4)
        #ax.plot(toplayer_x, toplayer_y, color='grey', marker=marker_f, linestyle='none', markersize=pointsize_f, transform=map_proj_f, zorder=4)
    if origpoint is None:
        print('No <origpoint> was found by get_map_lowfreq_var() function !')
    else:
        ax.plot(origpoint[0], origpoint[1], color='blue', marker='X', linestyle='none', markersize=2, transform=ccrs.PlateCarree(), zorder=5)
        #ax.plot(origpoint[0], origpoint[1], color='blue', marker='X', linestyle='none', markersize=2, transform=map_proj_f, zorder=5) 
    
    # # plot parallels and meridians
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='blue', alpha=0.5, linestyle='dotted', zorder=6)
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    
    # plot parallels at 30 and 70 degrees latitude
    # pdb.set_trace()
    if yy_f[0,0] > 0:
        parallels = [30-halfres_f, 70+halfres_f]
    elif yy_f[0,0] < 0:
        parallels = [-30+halfres_f, -70-halfres_f]
    else:
        raise ValueError('Unexpected value for <yy_f[0][0]> !')
    
    # for lat_f in parallels:
    #     pdb.set_trace()
    #     #ax.plot([-180, 180], [lat, lat], color='gray', linestyle='--', transform=ccrs.PlateCarree(), zorder=6)
    #     #ax.gridlines(xlocs=[], ylocs=[lat_f], color='gray', linestyle='solid', linewidth=1)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='solid', xlocs=[], ylocs=parallels, zorder=6)
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER

    if cbar_norm_f == None:
        cbar = plt.colorbar(image,orientation='vertical', shrink = 0.6)
    elif cbar_norm_f > 0:
        cbar = plt.colorbar(image,orientation='vertical', shrink = 0.6, boundaries=boundaries_f, ticks=np.arange(cbar_norm_f+1))
    else:
        raise ValueError("Unknown entry for <cbar_norm_f> !")

    cbar.set_label(cbarlabel_f, rotation=270, labelpad=+12, y=0.5, fontsize=titlesize_f)
    plt.title(title_f, fontsize=titlesize_f-1)
    plt.savefig(savename_f,dpi=dpival_f)
    plt.close('all')

def get_monthly_lwt_counts(nc_f,lwt_f):
    '''calculates monthly LWT counts from hourly instantaneous LWT type time series.'''
    #modify origional LWT time series containing 27 types to a binary absence (0) - occurrence (1) time series of the requested types only
    bin_array_f = np.zeros(nc_f.shape)
    tarwt_ind_f = np.where(nc_f.isin(lwt_f))
    bin_array_f[tarwt_ind_f] = 1
    arr_tarwts_f = xr.DataArray(data=bin_array_f,coords=[pd.DatetimeIndex(nc_f.time),nc_f.lon,nc_f.lat],dims=['time','lon','lat'],name='binary_lwt')
    #calculate monthly sums
    arr_tarwts_f = arr_tarwts_f.resample(time='1M').sum()
    days_per_month_f = arr_tarwts_f.time.dt.days_in_month
    if any(days_per_month_f < 28):
        raise Exception('ERROR: unexpected entry for <days_per_month_f> !')
    return(arr_tarwts_f,days_per_month_f)
    # arr_tarwts_f.close()
    # days_per_month_f.close()
    # nc_f.close()
    # del(nc_f,lwt_f,arr_tarwts_f,days_per_month_f,bin_array_f,tarwt_ind_f,bin_array_f)
    # gc.collect() #explicetly free memory

def get_seasonal_mean(xr_ds_f,months_f):
    '''calculates seasonal mean values for the data variables located in the xarray dataset <xr_ds_f>
    for the season defined in the <months_f> list, the latter containing intergers, e.g. [12,1,2] for the DJF season'''
    xr_ds_f = xr_ds_f.isel(time=xr_ds_f.time.dt.month.isin(months_f)) #get target months
    xr_ds_f = xr_ds_f.rolling(time=len(months_f)).sum() #rolling sum
    xr_ds_f = xr_ds_f.isel(time=xr_ds_f.time.dt.month == months_f[-1]) #get the accumulated values ending in the last month indicated in <months_f>
    return(xr_ds_f)
    xr_ds_f.close()
    del(xr_ds_f,months_f)

def map_polar_single_field(xr_arr,title,savename_nh,savename_sh,minval,maxval,dpival,colormap,titlesize,cbarlabel,cbar_norm_f=None):
    '''maps a single field without superimposed layer for the northern and southern hemsiphere separately using a polar stereographic projection'''
    xr_arr_nh = xr_arr.isel(lat=xr_arr.lat >= 0)
    xr_arr_sh = xr_arr.isel(lat=xr_arr.lat < 0)
    xx_nh, yy_nh = np.meshgrid(xr_arr_nh.lon.values,xr_arr_nh.lat.values)
    xx_sh, yy_sh = np.meshgrid(xr_arr_sh.lon.values,xr_arr_sh.lat.values)    
    if os.path.isdir(savedir) != True:
        os.makedirs(savedir)
    halfres = (xr_arr.lat.values[1]-xr_arr.lat.values[0])/2
    get_map_lowfreq_var(np.transpose(xr_arr_nh.values),xx_nh,yy_nh,minval,maxval,dpival,title,savename_nh,halfres,colormap,titlesize,cbarlabel,ccrs.NorthPolarStereo(),agree_ind=None,origpoint=None,cbar_norm_f=cbar_norm_f)
    get_map_lowfreq_var(np.transpose(xr_arr_sh.values),xx_sh,yy_sh,minval,maxval,dpival,title,savename_sh,halfres,colormap,titlesize,cbarlabel,ccrs.SouthPolarStereo(),agree_ind=None,origpoint=None,cbar_norm_f=cbar_norm_f)
    xr_arr.close()
    xr_arr_nh.close()
    xr_arr_sh.close()
    del(xr_arr,xr_arr_nh,xr_arr_sh)

def get_rpc(xr_mod_f,xr_mod_mean_f,xr_pearson_r_f,approach='Eade'):
    ''' Calculates the ratio of predictable components (RPC) following Eade et al. 2014, doi: 10.1002/2014GL061146 or Scaife et al. 2018, https://doi.org/10.1038/s41612-018-0038-4;
    <xr_mod_f> contains the memberwise time-series of the ensemble, <xr_mod_mean_f> contains the ensemble-mean time series averaged year-to-year over all members, and <xr_pearson_r_f>
    the Pearson correlation coefficient of the ensemble-mean time-series with the observations i.e. the observed predictable component; all the 3 objects are xarray DataArrays'''    
    if approach == 'Eade':
        ##following Eade et al. 2014, https://doi.org/10.1002%2F2014GL061146, equation 1
        var_sig_f = xr_mod_mean_f.var(dim='time') #calculate the variance of the ensemble-mean time series
        var_tot_f = xr_mod_f.var(dim='time').mean(dim='run_index') #calculates the mean of the individual members' variance
        pc_mod_f = np.sqrt(var_sig_f / var_tot_f) #modelled predictable component, equivalent to perfect skill or potential skill
        rpc_f = (xr_pearson_r_f / pc_mod_f).rename('rpc_eade') #calculate the RPC
        var_sig_f.close()
        var_tot_f.close()
        del(var_sig_f,var_tot_f)
    elif approach == 'Cottrell':
        # following Cottrell et al. 2023, https://doi.org/10.1002/asl.1212, equation 3 working with the correlation coefficient
        pc_mod_f = xs.pearson_r(xr_mod_f,xr_mod_mean_f,dim='time',skipna=True).rename('pearson_r').mean(dim='run_index') #calculate the correlation coefficient between the multi-model mean and each member and take the mean along all members obtaining the modelled predictable component
        rpc_f = (xr_pearson_r_f / pc_mod_f).rename('rpc_scaife') #calculate the RPC
    elif approach == 'Scaife':
        # following Scaife & Smith 2018, equation 5 working with explained variance
        rpc_f = np.sqrt(xr_pearson_r_f**2 / pc_mod_f**2).rename('rpc_scaife_expvar') #calculate the RPC
    else:
        raise Exception('ERROR: check entry for <approach> input parameter in the get_rpc() function !')
    
    return(rpc_f,pc_mod_f)
    xr_mod_f.close()
    xr_mod_mean_f.close()
    xr_person_f.close()
    rpc_f.close()
    pc_mod_f.close()
    del(xr_mod_f,xr_mod_mean_f,xr_pearson_r_f,rpc_f,pc_mod_f)

def pvalue_correlation_diff(pearson1, pearson2, n1, n2):
    '''caclulates the p-value for the difference in the correlation coefficients contained in the xarray DataArrays <pearson1> and <pearson2> 
    caclculated upon the sample sizes <n1> and <n2>, respectively, both being either numpy arrays, floats or integers; <pearson1> and <pearson2> must have the same dimensions;
    see https://stats.stackexchange.com/questions/278751/how-do-i-determine-whether-two-correlations-are-significantly-different'''
    # Fisher z transformation
    z1 = 0.5 * np.log((1 + pearson1) / (1 - pearson1))
    z2 = 0.5 * np.log((1 + pearson2) / (1 - pearson2))
    # # or equivalently
    # z1 = np.arctanh(pearson1)
    # z2 = np.arctanh(pearson2)

    # Standard error of the difference between correlations
    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    # Calculate z-score for the difference
    z_diff = (z1 - z2) / se_diff
    # Compute p-value
    p_value = 2 * (1 - norm.cdf(np.abs(z_diff)))
    p_value = xr.DataArray(p_value,coords=pearson1.coords,dims=pearson1.dims,name='p_value')
    return(p_value)
    pearson1.close()
    pearson2.close()
    p_value.close()
    del(pearson1,pearson2,p_value)

def draw_boxplot(xr_arr_f,xticks_f,tarwts_name_f,cbarlabel_f,savename_f,dpival_f,ymin_f,ymax_f,critval_f=None):
    '''draws a boxplot of the quantity provided in the xarray data array <xr_arr_f> with two dimensions containing
    1) the variable to be plot and 2) the grid-boxes in flattened format without the need of geographical reference information, i.e. not lat x lon or similar'''
    fig = plt.figure()
    plt.plot([-0.5,xticks_f[-1]+0.5],[0,0],linestyle='dotted',color='black',zorder=1) #plot 0 line
    if critval_f is not None:
        plt.plot([-0.5,xticks_f[-1]+0.5],[critval_f,critval_f],linestyle='dotted',color='red',zorder=2)
        plt.plot([-0.5,xticks_f[-1]+0.5],[critval_f*-1,critval_f*-1],linestyle='dotted',color='red',zorder=3)
    sns.boxplot(data=xr_arr_f.transpose('gridpoints','lwt'),zorder=4)
    plt.xticks(ticks=xticks_f,labels=tarwts_name_f)
    plt.xlabel('Lamb Weather Type')
    plt.ylim([ymin_f,ymax_f]) #y axis is centred around 0
    plt.ylabel(cbarlabel_f)
    #plt.grid(axis='y',linestyle='dotted')
    plt.savefig(savename_f,dpi=dpival_f)
    plt.close('all')
    del(fig)
    xr_arr_f.close()
    del(xr_arr_f,xticks_f,critval_f,tarwts_name_f,cbarlabel_f,savename_f,dpival_f)

def z2r(z_f):
    '''back-transformation of z-values provided in <z_f> to the Pearson correlation coefficient returned in <r_f> following Schönwiese 2006, page 178'''
    r_f = (np.exp(z_f)-np.exp(z_f*-1)) / (np.exp(z_f)+np.exp(z_f*-1))
    return(r_f)

def fisher_z_transform(r):
    '''performs a z-transformation of the Pearson correlation coefficient'''
    return(np.arctanh(r))

def z_test(r1, r2, n1, n2):
    '''performs a z-test '''
    z1 = fisher_z_transform(r1)
    z2 = fisher_z_transform(r2)
    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_diff = (z1 - z2) / se_diff
    p_value = 2 * (1 - norm.cdf(np.abs(z_diff))) # assuming a two-tailed test
    return(z_diff, p_value)

