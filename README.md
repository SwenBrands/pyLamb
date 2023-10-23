The pyLamb package is a set of Python scripts and functions made to
evaluate global climate models in terms of atmospheric circulation
patterns on multiple time-scales and also explores the low frequency 
variability seen in observations. To this end, the 2 main scripts 
<interpolate_xesmf.py> and <makecalcs_parallel.py> are provided that 
must be run sequentially in order to classify continuous 6-hourly 
sea-level pressure patterns into 27 discrete the Lamb Weather Types
(LWTs, Jones et al. 1993) at any grid-box in northern and southern Hemisphere
extratropics in reanalysis and GCM data. The package is designed to process netCDF
data from ESGF of virtually any GCM participating in CMIP5 and 6 and
additionally comes with an extensive metadata archive on these GCMs.
The scripts have the following tasks and should be run in the following
order

1. interpolate_xesmf.py: interpolates GCM or reanalysis data from the
grid provided by ESGF to a common 2.5 grid covering either the Northern
or Southern Hemisphere extratropics between 30 and 70 degress North or
South. Converts non-standard GCM calendars to a standard calendar and
keeps the original metadata provided by the ESGF files. Interpolation
methods from the xesmf module can be applied. Should be run with
6-hourly psl data as input. Save results in netCDF format.

2. makecalcs_parallel.py: assigns a single Lamb Weather Type to each
time instant and grid-box of the netCDF files generated in 1. The script
can be run in parallel mode, which significantly speeds it up. Saves
the results in netCDF format keeping the original metadata from the ESGF
files passed throug 1. With some small modifications can also calclate
the 5 circulaation inidices / intermediate variables of the LWT approach.

3. analysis_hist.py: is the main script to calculate several climatologcial
aspects of the point-wise LWT time series in the models and verify them
against reanalysis data. The focus is here put on verifying one member
per GCM, taking a large number of different GCMs into acccount. The
applied methods and metrics are explained in https://doi.org/10.5194/gmd-15-1375-2022
In addtion, GCM dependencies are estimated by means of error pattern
correlation as described in https://doi.org/10.1029/2022GL101446

4. analysis_ensemble.py: As 3, but for ensembles of a reduced number of GCMs

5. analysis_functions.py: Contains the functions used in the scripts

6. get_composites.py: Retrieves and plots average SLP composites for a given
dataset, location (or city) and LWT

7. get_corr.py: subscript of analysis_hist.py, used to calculate
correlations between atmosphere and ocean resolution paramters and
mean / median GCM performance

8. to be continued...

Within these scripts, a large number of input parameters can be set
that are commented to understand their meaning or purpose. This is
followed by a line stating "EXECUTE" from where on no futher user
options appear, i.e. the script is run with the parameters set before.
