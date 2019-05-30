"""
    This script plots zonal and meridional wind speed over a single Titan year.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
import numpy as np
import glob



def velu(ds, time):
    u = ds.ucomp.mean(dim='lon')
    u1 = (u[:, 12, 14] + u[:, 12, 15])/2 #SH Eq.
    u2 = (u[:, 12, 16] + u[:, 12, 17])/2 #NH Eq
    u3 = (u[:, 12, 0] + u[:, 12, 1] + u[:, 12, 2] + u[:, 12, 3] + u[:, 12, 4])/5 #SH Pole
    u4 = (u[:, 12, 27] + u[:, 12, 28] + u[:, 12, 29] + u[:, 12, 30] + u[:, 12, 31])/5 #NH Pole
    label = 'Zonal Wind (m s$^{-1}$)'
    maxV=0.035
    minV=-0.015
    
    return plots(time, u1, u2, u3, u4, label, maxV, minV)


def velv(ds, time):
    v = ds.vcomp.mean(dim='lon')
    v1 = (v[:, 12, 14] + v[:, 12, 15])/2 #SH Eq.
    v2 = (v[:, 12, 16] + v[:, 12, 17])/2 #NH Eq
    v3 = (v[:, 12, 0] + v[:, 12, 1] + v[:, 12, 2] + v[:, 12, 3] + v[:, 12, 4])/5 #SH Pole
    v4 = (v[:, 12, 27] + v[:, 12, 28] + v[:, 12, 29] + v[:, 12, 30] + v[:, 12, 31])/5 #NH Pole
    label = 'Meridional Wind (m s$^{-1}$)'
    maxV=0.01
    minV=-0.01
    
    return plots(time, v1, v2, v3, v4, label, maxV, minV)

def plots(time, y1, y2, y3, y4, label, maxV, minV):
    
       
    fig1 = plt.figure(figsize=(7.5,4.5))
    ax1 = fig1.add_subplot(111)    
    
    ax1.plot(time, y1, color='#2980B9', linewidth=3)
    ax1.plot(time, y2, color='#27AE60', linewidth=3)
    ax1.plot(time, y3, color='#2980B9', linewidth=3, linestyle='--')
    ax1.plot(time, y4, color='#27AE60', linewidth=3, linestyle='--')
    ax1.set_xlabel('Time (Titan Years)', fontsize = 'xx-large')
    
    ax1.set_ylabel(label, fontsize = 'xx-large')
    
    ax1.set_xlim(0, 1.0)
    ax1.xaxis.set_ticks_position('both')
    plt.xticks(np.arange(0,1.25,0.25), ('0', '0.25', '0.5', '0.75', '1'))
    ax1.set_ylim(minV, maxV)
    #plt.yticks([-0.005, 0, 0.005])
    ax1.tick_params(axis='y', labelsize = 'x-large', direction='in', which='both')
    ax1.tick_params(axis='x', labelsize = 'xx-large', direction='in', which='both')
    
    plt.legend(labels=('S. Eq', 'N. Eq', 'S. Polar', 'N. Polar'), loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=False, shadow=True, ncol=4, fontsize = 'xx-large')
    
    #the following lines mark the equinoxes and solstices
    ax1.plot([0.26,0.26], [minV, maxV], 'gainsboro')
    ax1.plot([0.52,0.52], [minV, maxV], 'gainsboro')
    ax1.plot([0.77,0.77], [minV, maxV], 'gainsboro')
    
    #ax2 = ax1.twinx()
    #ax2.plot(Tyears, r, color='k', linewidth=2, linestyle=':')
    #ax2.set_ylabel('Titan-Sun Distance (AU)', fontsize = 'xx-large')
    #ax2.yaxis.set_tick_params(labelsize = 'xx-large')
    #ax2.set_xlim(0, 1)
    #ax2.set_ylim(9, 10)
    #ax2.plot([-1,1], [9.5, 9.5], 'gainsboro')
    
#    plt.title('(a)', fontsize='xx-large', pad=10)
    plt.tight_layout()

if __name__ == '__main__': 
    files = sorted(glob.glob('grey_titan/year/run*/*.nc'))
    ds = xr.open_mfdataset(files, decode_times=False)
    
    tm = ds.coords['time'].data
    Tyears = tm/(15.9*673) - 6.41 #to just plot over a year
    rrsun = ds.rrsun[:,0]
    r = (1/np.sqrt(rrsun))*np.sqrt(1360/15.08) #not non-dimensional
    
    velu(ds, Tyears)
    velv(ds, Tyears)