"""
    Script for finding the indices for closest values to specific latitude, longitude and altitude inputs.
    Takes user inputs.
    Outputs the closest values in the .nc dataset plus the index of the valye in the lat, lon and pfull arrays.
    Best to run this script in a run000X directory.
"""

import xarray as xr
from bisect import bisect_left
import numpy as np

#the below values are for Titan
R = 290 #specific gas constant 
T = 93.65 #surface temperature K from A.Coustenis book
g = 1.354 #surface gravity from A.Coustenis book
p0 = 1467 #surface pressure in hPa
    
def closest(mylist, myval):
    if (myval > mylist[-1] or myval < mylist[0]):
        return False
    pos = bisect_left(mylist, myval)
    if pos == 0:
        return mylist[0]
    if pos == len(mylist):
        return mylist[-1]
    before = mylist[pos - 1]
    after = mylist[pos]
    if after - myval < myval - before:
        print('index=',pos)
        return after
    else:
        print('index=',pos)
        return before
    
ds = xr.open_dataset('atmos_monthly.nc', decode_times=False) 
p = ds.coords['pfull'].data
lat = ds.coords['lat'].data
lon = ds.coords['lon'].data

lonval = float(input("Input longitude to find: "))
latval = float(input("Input latitude to find: "))
zval = float(input("Input altitude to find: "))
pval = p0*np.exp((-1)*zval*(10**3)/((R*T/g))) #converts to pressure using z=-Hlog10(p/p0)

closestlat = print("Closest latitude:", closest(lat, latval))
closestlon = print("Closest longitude:", closest(lon, lonval))
newp = closest(p, pval)
closestp = print("Closest pressure:", newp)
closestz = print("Closest altitude:", (-1)*(R*T/g)*np.log((newp)/p0)/(10**3))

#can use the below line but it won't tell you the index
#print(ds.sel(lat=latval, lon=lonval, pfull=pval, method='nearest'))