# -*- coding: utf-8 -*-

''''This script plots composites maps for the 27 LWTs for a given location and time period'''

#load packages
import numpy as np
import pandas as pd
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import os
exec(open('analysis_functions.py').read())

#set input parameter
city = 'Weddell Sea'
tarmonths = list((1,2,3,4,5,6,7,8,9,10,11,12)) #target months
dist_lon = 20
dist_lat = 10
wtnr = 27 # 27 (full) or 11 (reduced LWT appraoch)
tarwt = 1
dpival = 300
gridres = 2.5
colormap = 'RdYlBu_r'
outformat = 'pdf'

gcm = 'cera20c'
run = 'm0'
experiment = '20c' #historical, 20c, amip, ssp245, ssp585
hemis = 'sh' #nh or sh

#execute ###############################################################################################
tarlat,tarlon = get_location(city)

if gcm == 'cera20c':
    timestep = '3h'
    taryears = [1901,2010] #start and end yeartaryears = [1979,2005] #start and end year
elif gcm == 'era5':
    timestep = '6h'
    taryears = [1979,2020] #start and end yeartaryears = [1979,2005] #start and end year
else:
    timestep = '6h'
    taryears = [1979,2005] #start and end year
    
wt_names = ['PA', 'DANE', 'DAE', 'DASE', 'DAS', 'DASW', 'DAW', 'DANW', 'DAN', 'PDNE', 'PDE', 'PDSE', 'PDS', 'PDSW', 'PDW', 'PDNW', 'PDN', 'PC', 'DCNE', 'DCE', 'DCSE', 'DCS', 'DCSW', 'DCW', 'DCNW', 'DCN', 'U']
lustre = os.getenv('LUSTRE')
store_slp = '/lustre/gmeteo/WORK/swen/datos/GCMData/'+gcm+'/'+timestep+'/'+experiment+'/'+run+'/psl_interp'
store_wt = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/results_v2/'+timestep+'/'+experiment+'/'+hemis

slp_file = store_slp+'/psl_interp_'+timestep+'rPlev_'+gcm+'_'+experiment+'_'+run+'_'+hemis+'.nc' #path to the raw slp data
wt_file = store_wt+'/wtseries_'+gcm+'_'+experiment+'_'+run+'_'+hemis+'_'+str(taryears[0])+'_'+str(taryears[1])+'.nc' #path to the LWT catalogues
#figs = lustre+'/datos/tareas/lamb_cmip5/figs/composite_means/'+gcm+'/'+run+'/'+hemis
figs = '/lustre/gmeteo/WORK/swen/datos/tareas/lamb_cmip5/figs'

tarlon = abs(-180-tarlon) # convert target longitude into 0 - 360 degrees format
slp = xr.open_dataset(slp_file)
res = np.gradient(slp.lon)[0] #resultion is assumed to be equal in latitude and longitude
lon = slp.lon.values
lat = slp.lat.values

#get the gridbox nearest to tarlon and tarlat
lon_center = slp.lon.sel(lon=tarlon,method='nearest').values
lat_center = slp.lat.sel(lat=tarlat,method='nearest').values

#find the indices for lon, lat and time you want to retrieve
lon_min = lon_center - dist_lon
lon_max = lon_center + dist_lon
lat_min = lat_center - dist_lat
lat_max = lat_center + dist_lat
lon_ind = np.where((lon >= lon_min) & (lon <= lon_max))[0]
lat_ind = np.where((lat >= lat_min-gridres) & (lat <= lat_max+gridres))[0]
dates_slp = pd.DatetimeIndex(slp.time.values)
year_ind_slp = np.where((dates_slp.year >= taryears[0]) & (dates_slp.year <= taryears[1]))[0]

#then retrieve the slp data on this grid and for these instances in time and close the nc file
slp_grid = slp.isel(lat=lat_ind,lon=lon_ind,time=year_ind_slp)
slp.close()

#load the LWT time series for the centerlat and target years obtained above
wt = xr.open_dataset(wt_file)
wt_center = wt.sel(lon=lon_center,lat=lat_center)
dates_wt = pd.DatetimeIndex(wt_center.time.values)
year_ind_wt = np.where((dates_wt.year >= taryears[0]) & (dates_wt.year <= taryears[1]))[0]
wt_center = wt_center.isel(time=year_ind_wt)
wt_val = wt_center.wtseries.values
#wt.close()

slp_grid['lon'] = slp_grid.lon-180
lon_center = np.array(lon_center-180.)
lon_step = np.linspace(lon_center-15.,lon_center+15.,4)
lat_step = np.linspace(lat_center-10.,lat_center+10.,5)

#überprüfen
coord_lon = np.concatenate((lon_step[1:3],lon_step,lon_step,lon_step,lon_step[1:3]))
coord_lat = np.concatenate((np.repeat(lat_step[-1],2),np.repeat(lat_step[3],4),np.repeat(lat_step[2],4),np.repeat(lat_step[1],4),np.repeat(lat_step[0],2)))

#plot composite slp map for each LWT
savedir_compo = figs+'/'+gcm+'/local/'+city+'/composites'
#create target directory if missing
if os.path.isdir(savedir_compo) != True:
    os.makedirs(savedir_compo)

nrtime = slp_grid.time.shape[0]
relfreq = np.zeros(wtnr)
wt_center_dti = wt_center.copy() #make a copy to work with pandas DatetimeIndex (dti)
wt_center_dti['time'] = pd.DatetimeIndex(wt_center_dti.time) #converts time vector to pandas DatetimeIndex
for ii in list(range(wtnr)):
    wt_ind = np.where(wt_val == ii+1)[0]
    relfreq[ii] = len(wt_ind)/nrtime*100
    slp_grid_mn = slp_grid.isel(time=wt_ind).mean(dim='time')/100 #transform Pa to hPa
    plotme = slp_grid_mn.psl.transpose()
    fig = plt.figure()
    ax = plotme.plot.contourf(subplot_kws=dict(projection=ccrs.Orthographic(lon_center, lat_center), facecolor="white"), levels = 15, cmap=colormap, transform=ccrs.PlateCarree())
    plt.plot(lon_center,lat_center,marker='o',color='white')
    for pp in list(range(len(coord_lon))):
       plt.plot(coord_lon[pp],coord_lat[pp],marker='x',color='white',transform=ccrs.Orthographic(coord_lon[pp], coord_lat[pp]))
    #plt.plot(coord_lon,coord_lat,'ow',transform=ccrs.Orthographic())
        
    ax.axes.add_feature(cartopy.feature.BORDERS)
    ax.axes.coastlines()
    plt.title(str(wt_names[ii])+', '+str(np.round(relfreq[ii],1))+'%, '+gcm+', '+run+', '+str(taryears[0])+'-'+str(taryears[1]))
    savename = savedir_compo+'/slp_composites_LWT'+str(ii+1)+'_'+gcm+'_'+run+'_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
    plt.savefig(savename,dpi=dpival)
    plt.close('all')

#get time series of yearly WT counts
tarwt_ind = np.where(wt_center_dti.wtseries.values == tarwt)
tarwt_out = np.where(wt_center_dti.wtseries.values != tarwt)
wt_center_dti['wtseries'][tarwt_ind] = 1
wt_center_dti['wtseries'][tarwt_out] = 0
wt_center_yr = wt_center_dti.groupby('time.year').sum('time')
fig = plt.figure()
plt.plot(wt_center_yr.year,wt_center_yr.wtseries)
plt.xlabel('year')
plt.ylabel(timestep+' occurrence frequency')
plt.ylim([wt_center_yr.wtseries.values.min()-10,wt_center_yr.wtseries.values.max()+10])
plt.title('LWT'+str(tarwt)+', '+gcm+', '+run+', '+str(taryears[0])+'-'+str(taryears[1]))

savedir_ts = figs+'/'+gcm+'/local/'+city+'/composites/timeseries'
#create target directory if missing
if os.path.isdir(savedir_ts) != True:
    os.makedirs(savedir_ts)

savename = savedir_ts+'/timeseries_LWT'+str(tarwt)+'_'+gcm+'_'+run+'_'+str(taryears[0])+'_'+str(taryears[1])+'.'+outformat
plt.savefig(savename,dpi=dpival)
plt.close('all')
wt_center_dti.close()
wt.close()
