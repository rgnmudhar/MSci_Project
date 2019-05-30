"""
    This script plots temperature, zonal wind and incoming solar radiation as a function of time.
    Can choose to plot for T21 or T42 resolution.
    Plots the equatorial averages for temperature and wind by taking the mean of the two latitudes either side of 0.
    Plots the global mean for sw radiation.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob

def data(ds, r, time):
    resolution = input("Is the data a) T21 or b) T42 resolution (type a or b): ")

    if resolution == "a":
        t1 = (ds.t_surf.mean(dim='lon')[:,15]+ds.t_surf.mean(dim='lon')[:,16])/2 #0km
        t2 = (ds.temp.mean(dim='lon')[:,12,15]+ds.temp.mean(dim='lon')[:,12,16])/2 #22km
        t3 = (ds.temp.mean(dim='lon')[:,10,15]+ds.temp.mean(dim='lon')[:,10,16])/2 #41km
        
        sw = ds.swdn_toa.mean(dim='lat').mean(dim='lon').data
        
        u1 = (ds.ucomp.mean(dim='lon')[:, 24, 16] + ds.ucomp.mean(dim='lon')[:, 24, 15])/2 
        u2 = (ds.ucomp.mean(dim='lon')[:, 12, 16] + ds.ucomp.mean(dim='lon')[:, 12, 15])/2 
        u3 = (ds.ucomp.mean(dim='lon')[:, 10, 16] + ds.ucomp.mean(dim='lon')[:, 10, 15])/2 
        
        plot_sw(time, r, sw)
        plot_x(time, r, t1, t2, t3, 't')
        plot_x(time, r, u1, u2, u3, 'u')
        
    elif resolution == "b":
        t1 = (ds.t_surf.mean(dim='lon')[:,31]+ds.t_surf.mean(dim='lon')[:,32])/2
        t2 = (ds.temp.mean(dim='lon')[:,12,31]+ds.temp.mean(dim='lon')[:,12,32])/2
        t3 = (ds.temp.mean(dim='lon')[:,10,31]+ds.temp.mean(dim='lon')[:,10,32])/2 
        
        sw = ds.swdn_toa.mean(dim='lat').mean(dim='lon').data
        
        u1 = (ds.ucomp.mean(dim='lon')[:, 24, 31] + ds.ucomp.mean(dim='lon')[:, 24, 32])/2
        u2 = (ds.ucomp.mean(dim='lon')[:, 12, 31] + ds.ucomp.mean(dim='lon')[:, 12, 32])/2
        u3 = (ds.ucomp.mean(dim='lon')[:, 10, 31] + ds.ucomp.mean(dim='lon')[:, 10, 32])/2
        
        plot_sw(time, r, sw)
        plot_x(time, r, t1, t2, t3, 't')
        plot_x(time, r, u1, u2, u3, 'u')

def plot_sw(time, r, sw):
    """Plots incoming solar radiation and Titan-Sun distance over time"""
    fig1 = plt.figure(figsize=(7.5,6.5))
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()
    
    ax1.plot(time, r, color='k', linewidth=1, linestyle=':')
    ax1.set_ylabel('Titan-Sun Distance (AU)', fontsize='xx-large')
    ax1.set_xlabel('Time (Titan Years)', fontsize='xx-large')
    ax1.set_xlim(0, max(time))
    
    ax2.plot(time, sw, color='#9B59B6', linewidth=3)
    ax2.set_xlabel('Time (Titan Years)', fontsize='xx-large')
    ax2.set_xlim(0, max(time))
    ax2.set_ylabel('Global Mean Incoming SW (W m$^{-2}$)', color='#9B59B6', fontsize='xx-large')
                   
    ax2.tick_params(axis='y', labelsize = 'xx-large', direction='in', colors='#9B59B6', which='both')
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    ax2.tick_params(axis='x', labelsize = 'xx-large', direction='in', which='both')
#    plt.title('(a)', fontsize='xx-large', pad=10)
    return plt.tight_layout()

def plot_x(time, r, x1, x2, x3, x_label):
    """Plots either temperature or zonal wind and Titan-Sun distance over time"""
    
    if x_label == 't':
        x = 'Equatorial Mean Temperature (K)'
    elif x_label == 'u':
        x = 'Equatorial Mean Zonal Wind (m/s)'
    
    fig1 = plt.figure(figsize=(7.5,6.5))
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()
    
    ax1.plot(time, r, color='k', linewidth=1, linestyle=':')
    ax1.set_ylabel('Titan-Sun Distance (AU)', fontsize='xx-large')
    ax1.set_xlabel('Time (Titan Years)', fontsize='xx-large')
    ax1.set_xlim(0, max(time))
    
    ax2.plot(time, x1, color='#2980B9', linewidth=3)
    ax2.plot(time, x2, color='#27AE60', linewidth=3)
    ax2.plot(time, x3, color='#C0392B', linewidth=3)
    ax2.set_xlabel('Time (Titan Years)', fontsize='xx-large')
    ax2.set_xlim(0, max(time))
    ax2.set_ylabel(x, fontsize='xx-large')
    
    leg = ax2.legend(labels=('Surface', '22 km', '41 km'), loc='upper center' , bbox_to_anchor=(0.5, -0.07),
                     fancybox=False, shadow=True, ncol=5, fontsize='xx-large')
    
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    ax2.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
#    plt.title('(b)', fontsize='xx-large', pad=10)
    return plt.tight_layout()

if __name__ == '__main__':
    files = sorted(glob.glob('titan_T21_3/run*/*.nc'))
    ds = xr.open_mfdataset(files, decode_times=False)
    tm = ds.coords['time'].data
    Tyears = tm/(15.9*673)
    rrsun = ds.rrsun[:,0]
    r = (1/np.sqrt(rrsun))*np.sqrt(1360/15.078) #not non-dimensional
    data(ds, r, Tyears)
