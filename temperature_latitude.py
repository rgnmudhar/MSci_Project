"""
    This script plots temperature against latitude.
    Note that I separated my data into separate folders for each season.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
import numpy as np
import glob

files1 = sorted(glob.glob('titan_T21_3/spring/run*/*.nc'))
ds1 = xr.open_mfdataset(files1, decode_times=False)
lat = ds1.coords['lat'].data
latitude = ('Latitude ('+chr(176)+')')

def temp(ds):
    """Temperatures at 3 different altitudes"""
    T = ds.temp.mean(dim='time').mean(dim='lon')
    T1 = ds.t_surf.mean(dim='time').mean(dim='lon') #0km
    T2 = T[10,:] #22 km
    T3 = T[4,:] #114 km
    return T1, T2, T3

def plots(ds):
    """Plots the temperatures against latitude"""
    
    lat = ds.coords['lat'].data
    T1, T2, T3 = temp(ds)

    fig1 = plt.figure(figsize=(7.5,6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(lat, T1, color='#2980B9', linewidth=3)
    ax1.set_xlabel('Latitude', fontsize = 'xx-large')
    ax1.set_ylabel('Temperature in Stratosphere (K)', fontsize = 'xx-large', color='#2980B9')
    ax1.tick_params(axis='y', labelsize = 'xx-large', colors='#2980B9')
    plt.xticks([-86, -43, 0, 43, 86], ['90S', '45S', '0', '45N', '90N'])
    ax1.set_xlim(-86, 86)
    ax1.tick_params(axis='both', labelsize = 'xx-large', which='both', direction='in')
    
    ax2=plt.twinx(ax1)
    ax2.plot(lat, T3, color='#E67E22', linewidth=3, linestyle=':')
    ax2.set_ylabel('Surface Temperature (K)', fontsize = 'xx-large', color='#E67E22')
    ax2.tick_params(axis='y', labelsize = 'x-large', colors='#E67E22', which='both', direction='in')
    ax2.tick_params(axis='x', labelsize = 'xx-large', which='both', direction='in')
    
    plt.title('Spring', fontsize='xx-large', pad=10) #spring
    plt.tight_layout()

if __name__ == '__main__': 
    files = sorted(glob.glob('titan_T21_3/spring/run*/*.nc'))
    ds = xr.open_mfdataset(files, decode_times=False)
    
    plots(ds)
