# -*- coding: utf-8 -*-

#initialize lists containing the metadata for each model, which is assigned below in <get_historical_metadata,py>
mrun = [' '] * len(model) #run specifications
doi = [' ']  * len(model) #dois
atmos = [' ']  * len(model) #GCM
surface = [' ']  * len(model) #land surface model
ocean = [' ']  * len(model) #OGCM
seaice = [' ']  * len(model) #sea-ice model
aerosols = [' ']  * len(model) #aerosols
chemistry = [' ']  * len(model) #atmospheric chemistry
obgc = [' ']  * len(model) #ocean biogeochemistry
landice = [' ']  * len(model) #ice sheets
complexity = [' '] * len(model) # model complexity code complexity, string
addinfo = [' ']  * len(model) #additional info
family = [' '] * len(model) #complexity family, currently "esm" or "gcm"
cmip = np.zeros(len(model)) #cmip generation, 5 or 6
rgb = [' '] * len(model) #rgb code for each model family
marker = [' '] * len(model) #marker for each model of a given family
latres_atm = [' '] * len(model) #number of meridional grid-boxes
lonres_atm = [' '] * len(model) #number of zonal grid-boxes at the equator (due to the presence of Gaussian reduced grids)
lev_atm = [' '] * len(model) #number of vertical levels in the atmosphere
latres_oc = [' '] * len(model) #as above, but for the ocean
lonres_oc = [' '] * len(model) # "
lev_oc = [' '] * len(model) # "
ecs = [' '] * len(model) #equilibrium climate sensitivity for each model
tcr = [' '] * len(model) #transient climate response for each model
experiment = [experiment] * len(model) #experiment type, "historical" in any case
