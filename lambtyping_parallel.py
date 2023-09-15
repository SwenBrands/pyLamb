# -*- coding: utf-8 -*-

"""

#Calculate all 27 Lamb Weather types following Jones et al. 2013, Int J Climatol, doi:10.1002/joc.3498 from an Xarray DataArray with coordinates time, lon and lat
#this functions works with the xarray and numpy packages, so ensure that these are loaded. The script only works for the northern Hemisphere.

#author: Swen Brands, swen.brands@gmail.com

# INPUT parameters are:
#1. dataset: xarray DataArray containing instantaneous mean-sea level pressure values in Pa on the coordinates time x lon x lat.
#   By definition of the Lamb Weather Typing (LWT) method, the data should come on a regular latitude-longitude grid with equal grid spacing along latitudes and longitudes (e.g. or 2.5 degrees).
#   This means that the SLP data from the GCM or reanlaysis saved in <dataset> should be interpolated to this resolution in a previous working step
#2. lons: longitudes of the coodinate system used by the classification scheme, keep a distance of 10 degrees.
#4. lats: respecitve latitudes, keep a distance of 5 degrees.
# 
# lons and lats must be in the following order (from left to right, hereafter referred to as the "Lamb cross")
# the grid box distances (10 deg in longitude and 5 deg in latitude) MUST be preserved##             
#
#                        North
#
#                       00    01
#                 02    03    04    05
#  West           06    07    08    09      East
#                 10    11    12    13
#                       14    15
#
#                       South

#5. tardates, can be optionally included: Pandas framework containing the list of dates to be loaded

# OUTPUT parameters are:
# 1.  wtseries: numpy array containging discrete values in the range of 1 to 27, one per time step, defining the circulation type at that moment in time
# 2.  wtnames: list containing the names of the 27 types
# 3.  wtcodes: list containing the integer code of the 27 types
# 4.  XX: the output data; numpy array containing SLP data at the grid-boxes nearest to the locations indicated in <lons> and <lats>.
# 5.  lon_real and lat_real: longitudes and latitudes as saved in <dataset>, should be identical to <lons> and <lats>; if they are not, the nearest neighbour is searched. 

@Author: Swen Brands, swen.brands@gmail.com
"""

#def lambtyping(dataset,lons1,lats1,tardates): #this version was able to cut out target dates with xr.sel, buit did not work with Hadley center models, try to solve this with isel in the future
def lambtyping_parallel(dataset,lons_lamb_0,lats_lamb_0,tarres,i,j,hemis):
    #default paramter values for Lamb typing
    uclim = 6;
    vortlim = 0; #default 0
    flowlim = 2; #default 2
    
    ##get grid-box distances in longitude and latitude
    lon_dist = 10.
    lat_dist = 5.
    
    #construct the coordinate system
    lons1 = lons_lamb_0 + i*tarres
    center_lons_step = (lons1[1] + lons1[0]) / 2.
    if hemis == 'nh':
        lats1 = lats_lamb_0 + j*tarres
    elif hemis == 'sh':
        lats1 = lats_lamb_0 - j*tarres
    else:
        raise Exception('ERROR: check entry for <hemis>')    
    center_lats_step = lats1[6]
    #print('INFO: Applying LWT approach at '+str(center_lons_step) +' longitude and '+str(center_lats_step)+' latitude....')

    #get lons and lats as saved in the netcdf file
    dataset_full = dataset.sel(lon = lons1, lat = lats1, method = 'nearest')
    lon_real = dataset_full.lon
    lat_real = dataset_full.lat
    
    #XX = np.zeros((tardates.shape[0],len(lons1)))
    XX = np.zeros((dataset.time.shape[0],len(lons1)))
    for ii in range(len(lons1)):
        XX[:,ii] = dataset.psl.sel(lon=lons1[ii],lat=lats1[ii], method='nearest')/100
        #XX[:,ii] = dataset.psl.sel(lon=lons1[ii],lat=lats1[ii], method='nearest', time=tardates)/100
 
    #follow Jones et al. 2012, Int. J. Climatol., last page
    centerlat = np.median(lats1)

    sfconst = 1./np.cos(np.deg2rad(centerlat)) #RICHTIG DIVIDIERT??
    zwconst1 = np.sin(np.deg2rad(centerlat))/np.sin(np.deg2rad(centerlat-lat_dist))
    zwconst2 = np.sin(np.deg2rad(centerlat))/np.sin(np.deg2rad(centerlat+lat_dist))
    zsconst = 1./(2*np.cos(np.deg2rad(centerlat))**2)

    #from the original FORTRAN code from Colin Harpham, CRU
    if centerlat > 0:
        w = 0.5*(XX[:,11] + XX[:,12]) - 0.5*(XX[:,3] + XX[:,4]) #zonal pressure gradient
        s = sfconst*(0.25*(XX[:,4] + 2*XX[:,8] + XX[:,12]) - 0.25*(XX[:,3] + 2*XX[:,7] + XX[:,11])) #meridional pressure gradient
    elif centerlat < 0:
        w = 0.5*(XX[:,3] + XX[:,4]) - 0.5*(XX[:,11] + XX[:,12]) #zonal pressure gradient
        s = sfconst*(0.25*(XX[:,3] + 2*XX[:,7] + XX[:,11]) - 0.25*(XX[:,4] + 2*XX[:,8] + XX[:,12]))
    elif centerlat == 0:
        raise Exception('ERROR: LWT are not defined at the Ecuator')
    else:
        raise Exception('ERROR: unknown entry for <centerlat>')

    dirdeg = np.zeros(w.shape[0])
    d = np.copy(dirdeg)
    ind1 = np.where(np.abs(w) > 0)
    ind2 = np.where(np.logical_and(np.abs(w) == 0, s >= 0))
    ind3 = np.where(np.logical_and(np.abs(w) == 0, s < 0))
    dirdeg[ind1] = np.rad2deg(np.arctan(s[ind1]/w[ind1]));
    dirdeg[ind2] = 90
    dirdeg[ind3] = -90

    #Definde directional quadrants
    indsw = np.where(np.logical_and(s > 0, w > 0))
    indse = np.where(np.logical_and(s > 0, w <= 0))
    indnw = np.where(np.logical_and(s <= 0, w > 0)) 
    indne = np.where(np.logical_and(s <= 0, w <= 0))
    d[indsw] =  270 - dirdeg[indsw]; #SW quadrant
    d[indse] =  90 - dirdeg[indse]; #SE quadrant
    d[indnw] =  270 - dirdeg[indnw]; #NW quadrant
    d[indne] =  90 - dirdeg[indne] #NE quadrant

    #southerly and westerly shear vorticity
    if centerlat > 0:
        zw = zwconst1*(0.5*(XX[:,14] + XX[:,15]) - 0.5*(XX[:,7] + XX[:,8])) - zwconst2*(0.5*(XX[:,7] + XX[:,8]) - 0.5*(XX[:,0] + XX[:,1]));
        zs = zsconst*(0.25*(XX[:,5] + 2*XX[:,9] + XX[:,13]) - 0.25*(XX[:,4] + 2*XX[:,8] + XX[:,12]) - 0.25*(XX[:,3] + 2*XX[:,7] + XX[:,11]) + 0.25*(XX[:,2] + 2*XX[:,6] + XX[:,10]));
    elif centerlat < 0:
        zw = zwconst2*(0.5*(XX[:,0] + XX[:,1]) - 0.5*(XX[:,7] + XX[:,8])) - zwconst1*(0.5*(XX[:,7] + XX[:,8]) - 0.5*(XX[:,14] + XX[:,15]));
        zs = zsconst*(0.25*(XX[:,2] + 2*XX[:,6] + XX[:,10]) - 0.25*(XX[:,3] + 2*XX[:,7] + XX[:,11]) - 0.25*(XX[:,4] + 2*XX[:,8] + XX[:,12]) + 0.25*(XX[:,5] + 2*XX[:,9] + XX[:,13]));
    elif centerlat == 0:
        raise Exception('ERROR: LWT are not defined at the Ecuator')
    else:
        raise Exception('ERROR: unknown entry for <centerlat>')

    #total shear vorticity
    z = zw+zs; #total shear vorticity
    f = np.sqrt(w**2+s**2); #resultant flow
    guk = np.sqrt(f*f+0.25*z*z);

    #define direction sectors form 1 to 8, definition like on http://www.cru.uea.ac.uk/cru/data/hulme/uk/lamb.htm
    neind = np.where(np.logical_and(d > 22.5, d <= 67.5)) #NE sector
    eind = np.where(np.logical_and(d > 67.5, d <= 112.5)) #E
    seind = np.where(np.logical_and(d > 112.5, d <= 157.5)) #SE
    sind = np.where(np.logical_and(d > 157.5, d <= 202.5)) #S
    swind = np.where(np.logical_and(d > 202.5, d <= 247.5)) #SW
    wind = np.where(np.logical_and(d > 247.5, d <= 292.5)) #W
    nwind = np.where(np.logical_and(d > 292.5, d <= 337.5)) #NW
    nind = np.where(np.logical_or(d > 337.5, d <= 22.5)) #N

    d[neind] = 10; d[eind] = 11; d[seind] = 12; d[sind] = 13
    d[swind] = 14; d[wind] = 15; d[nwind] = 16; d[nind] = 17

    pd = np.where(np.absolute(z) < f); #purely directional type
    pcyc = np.where(np.logical_and(np.absolute(z) >= flowlim*f, z >= vortlim)) #purely cyclonic type
    pant = np.where(np.logical_and(np.absolute(z) >= flowlim*f, z < vortlim)) #purely anticyclonic type
    hyb = np.where(np.logical_and(np.absolute(z) >= f, np.absolute(z) < flowlim*f)) #hybrid type
    unclass = np.where(np.logical_and(f < uclim, np.absolute(z) < uclim)) #unclassified type, see Jones et al 1993, Int J Climatol

    #define hybrid types
    hybne = np.intersect1d(hyb,np.where(d==10))
    hybe = np.intersect1d(hyb,np.where(d==11))
    hybse = np.intersect1d(hyb,np.where(d==12))
    hybs = np.intersect1d(hyb,np.where(d==13))
    hybsw = np.intersect1d(hyb,np.where(d==14))
    hybw = np.intersect1d(hyb,np.where(d==15))
    hybnw = np.intersect1d(hyb,np.where(d==16))
    hybn = np.intersect1d(hyb,np.where(d==17))

    zminus = np.where(z < 0); #anticylonic
    zplus = np.where(z >= 0); #cyclonic
    hybant = np.intersect1d(hyb,zminus);
    hybcyc = np.intersect1d(hyb,zplus);

    # Define discrete wt series, codes similar to http://www.cru.uea.ac.uk/cru/data/hulme/uk/lamb.htm
    wtseries = np.zeros(d.shape[0])
    wtseries[pd] = d[pd] #purely directional
    wtseries[pant] = 1 #purely anticyclonic
    wtseries[pcyc] = 18 #purely cyclonic
    #directional anticyclonic
    wtseries[np.intersect1d(hybant,hybne)] = 2
    wtseries[np.intersect1d(hybant,hybe)] = 3
    wtseries[np.intersect1d(hybant,hybse)] = 4
    wtseries[np.intersect1d(hybant,hybs)] = 5
    wtseries[np.intersect1d(hybant,hybsw)] = 6
    wtseries[np.intersect1d(hybant,hybw)] = 7
    wtseries[np.intersect1d(hybant,hybnw)] = 8
    wtseries[np.intersect1d(hybant,hybn)] = 9
    #mixed cyclonic
    wtseries[np.intersect1d(hybcyc,hybne)] = 19
    wtseries[np.intersect1d(hybcyc,hybe)] = 20
    wtseries[np.intersect1d(hybcyc,hybse)] = 21
    wtseries[np.intersect1d(hybcyc,hybs)] = 22
    wtseries[np.intersect1d(hybcyc,hybsw)] = 23
    wtseries[np.intersect1d(hybcyc,hybw)] = 24
    wtseries[np.intersect1d(hybcyc,hybnw)] = 25
    wtseries[np.intersect1d(hybcyc,hybn)] = 26
    wtseries[unclass] = 27
    
    #generate list with the name of each class
    wtnames = ['PA','DANE','DAE','DASE','DAS','DASW','DAW','DANW','DAN','PDNE','PDE','PDSE','PDS','PDSW','PDW','PDNW','PDN','PC','DCNE','DCE','DCSE','DCS','DCSW','DCW','DCNW','DCN','U']
    wtcode = list(range(1,28))
    #define output variables
    return(wtseries,wtnames,wtcode,w,s,zw,zs,z,f,XX,lon_real,lat_real,dirdeg,center_lons_step,center_lats_step)

