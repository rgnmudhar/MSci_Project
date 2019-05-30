""" 
    This script attempts to plot the Tropopause Height as a function of latitude.
    Doesn't show anything for Titan so unsure if it's really working. 
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob

def T(ds):
    ''' Take mean of average zonal temperature by taking averages along time and longitude dimensions '''
    
    T_avg = ds.temp.mean(dim='time').mean(dim='lon')
    
    return T_avg

def altitude(p):
    
    R = 290 #specific gas constant 
    T = 93.65 #surface temperature K from A.Coustenis book
    g = 1.354 #surface gravity from A.Coustenis book
    p0 = 1467 #surface pressure in hPa
    
    z = np.empty_like(p)
    
    for i in range(p.shape[0]):
        z[i] = (-1)*(R*T/g)*np.log((p[i])/p0)/(10**3)
        
    # Make into an xarray DataArray
    z_xr = xr.DataArray(z, coords=[z], dims=['pfull'])
    z_xr.attrs['units'] = 'km'
    
    #below is the inverse of the calculation
    #p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
    
    return z_xr

def troposphere(ds, p, lat):
    """ Restrict data to troposphere below 50 km """
    
    z = altitude(p)
    temp = T(ds)
    
    alt = []
    troposphere_temp = []
    
    
    for i in range(z.shape[0]):
        if z[i] <= 50:
            troposphere_temp.append(temp[i])
            alt.append(z[i])
    
    alt_xr = xr.DataArray(alt, coords=[alt], dims=['pfull'])
    alt_xr.attrs['units']='km'        
    
    temp_xr = xr.DataArray(troposphere_temp, coords=[alt_xr, lat], dims=['alt', 'lat'])
    temp_xr.attrs['units'] = 'K'
    
    return temp_xr

def tropopause(ds, p, lat):
    """Identify the tropopause temperature and height"""
    
    temp = troposphere(ds, p, lat)
    temp_at_trop = []
    z_trop = []
    
    for i in range(temp.shape[1]):
        min_temp = np.amin(temp[:,i])
        temp_at_trop.append(min_temp)
        index = np.where(temp[:,i] == min_temp)
        z_trop.append(temp[:,i].alt[index])
        
#    trop_temp_xr = xr.DataArray(temp_at_trop, coords=[lat], dims=['lat'])
#    trop_temp_xr.attrs['units'] = 'K'

    return z_trop

def plot(ds, p, lat):
    """Plot of tropopause height with latitude """
    
    alt = tropopause(ds, p, lat)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(lat, alt, color='#2980B9', linewidth=3)
    ax1.set_xlabel('Latitude (degrees)', fontsize='xx-large')
    ax1.set_xlim(min(lat), max(lat))
    ax1.set_ylabel('Tropopause height (km)', fontsize='xx-large')
    ax1.xaxis.set_tick_params(labelsize = 'xx-large')
    ax1.yaxis.set_tick_params(labelsize = 'xx-large')

    return plt.show()

if __name__ == '__main__': 
    files = sorted(glob.glob('grey_titan/run*/*.nc'))
    ds = xr.open_mfdataset(files, decode_times=False) #multiple files = mf, data for a year (12 months)
    lat = ds.coords['lat']
    p = ds.coords['pfull']
    
    #plot(ds, p, lat)
    
    plot(ds, p, lat)