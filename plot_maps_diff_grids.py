""" 
    This script reads in the .nc file and plots the regridded data
"""

import pandas as pd
import numpy as np
#from mpl_toolkits.basemap import Basemap
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator

def regrid():
    ds = Dataset('titan_land_t42.nc')
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    top = ds.variables['zsurf'][:]
    lsmask = ds.variables['land_mask'][:]
    
    spacing=30
    majorLocator=MultipleLocator(spacing)
    
    fig1 = plt.figure(figsize=(12,6))
    ax1 = fig1.add_subplot(111)
    topomap = plt.contourf(lons, lats, top, levels=21, cmap='copper')
    fig1.colorbar(topomap, pad=0.2).set_label(label='Topography (m)', size='xx-large')
    plt.contour(lons, lats, lsmask, levels=0, colors='white')
    plt.xlabel('Longitude ('+chr(176)+')', fontsize='xx-large')
    plt.xticks(np.arange(0, 360, 30))
    plt.ylabel('Latitude', fontsize='xx-large')
    plt.yticks(np.arange(-90, 120, 30), ['90S', '60S', '30S', '0', '30N', '60N', '90N'])
    ax1.xaxis.set_tick_params(labelsize = 'xx-large')
    ax1.yaxis.set_tick_params(labelsize = 'xx-large')
    ax1.xaxis.set_major_locator(majorLocator)
    plt.grid(True, which='major', alpha=0.3)
    #plt.title('Titan Topography and Land-Sea Mask (64x128 Grid)', fontsize='xx-large')
    plt.tight_layout()
    
#    fig2 = plt.figure()
#    plt.contourf(lons, lats, lsmask, levels = 1, cmap='RdBu_r')
#    plt.colorbar()
#    plt.xlabel('Longitude')
#    plt.ylabel('Latitude')
#    plt.title('Titan Land-Sea Mask (32x64 Grid)')
    
    return plt.show()

def original():
    filed = 'equi_4PPD_90N90S_geoid_removed.txt'
    filed2 = 'liquids_mask_4PPD_90N90S.txt'
    
    missing_data = [0.125,    0.125,   -422.6176]
    missing_data2 = [ 0.125,    0.125,  0.0]
    
    data = pd.read_csv(filed,delim_whitespace=True)
    data2 = pd.read_csv(filed2,delim_whitespace=True) 
    
    all_data = np.zeros((1036800,3))
    all_data[0,:]=missing_data
    all_data[1:,:] = data.values
    
    all_data2 = np.zeros((1036800,3))
    all_data2[0,:]=missing_data2
    all_data2[1:,:] = data2.values
    
    lons = np.zeros((1440))
    lats = np.zeros((720))
    top = np.zeros((720,1440))
    lsmask = np.zeros((720,1440))
    
    for count2 in np.arange(720):
        lats[count2] = -1*(all_data[(count2*1440)][0]-90)
        for count in np.arange(1440):
            #print(count + (count2*1440))
            lons[count] = all_data[count][1]
            
            top[count2,count] = all_data[count + (count2*1440)][2]
            lsmask[count2,count] = all_data2[count + (count2*1440)][2]
            
    spacing=30
    majorLocator=MultipleLocator(spacing)
    
    fig3 = plt.figure(figsize=(12,6))
    ax3 = fig3.add_subplot(111)
    topomap = plt.contourf(lons, lats, top, levels=21, cmap='copper')
    fig3.colorbar(topomap, pad=0.01).set_label(label='Topography (m)', size='xx-large')
    plt.contour(lons, lats, lsmask, levels=0, colors='white')
    plt.xlabel('Longitude ('+chr(176)+')', fontsize='xx-large')
    plt.xticks(np.arange(0, 360, 30))
    plt.ylabel('Latitude', fontsize='xx-large')
    plt.yticks(np.arange(-90, 120, 30), ['90S', '60S', '30S', '0', '30N', '60N', '90N'])
    ax3.xaxis.set_tick_params(labelsize = 'xx-large')
    ax3.yaxis.set_tick_params(labelsize = 'xx-large')
    ax3.xaxis.set_major_locator(majorLocator)
    plt.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    #plt.title('Titan Topography and Land-Sea Mask (720x1440 Grid)')
        
    return plt.show()

original()
regrid()