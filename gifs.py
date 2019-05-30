"""
    This script makes gifs of zonal temperature, wind speed and msf.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import xarray as xr
import numpy as np
import glob
import imageio
import os

def T(ds, z, lat):
    ''' Take mean of average zonal temperature by taking averages along time and longitude dimensions '''
    T_anm = ds.temp.mean(dim='time').mean(dim='lon').data 
    #ds.temp.mean(dim='time').mean(dim='lon')[13]
    T_xr = xr.DataArray(T_anm, coords=[z,lat], dims=['pfull','lat']) #xarray DataArray
    T_xr.attrs['units']='K'
    
    return T_xr

def uz(ds, z, lat):
    ''' Take annual mean of zonal wind speed by taking a mean along time and longitude dimensions '''
    u = ds.ucomp
    u_anm = u.mean(dim='time').mean(dim='lon').data 
    uz_xr = xr.DataArray(u_anm, coords=[z,lat], dims=['pfull','lat']) 
    uz_xr.attrs['units']='m/s'
    
    return uz_xr

def calc_streamfn(v, p, lat):
    '''Calculates the meridional streamfunction from v wind.

    Parameters
    ----------
        vz_xr: an xarray DataArray of form [pressure levs, latitudes]

    Returns
    -------
        psi_xr: xarray DataArray of meridional mass streamfunction, units of kg/s
    '''
    radius = 2575000.
    g = 1.354 #from A. Coustenis book
    coeff = (2*np.pi*radius)/g

    psi = np.empty_like(v)
    # Do the integration
    for ilat in range(lat.shape[0]):
        psi[0,ilat] = coeff*np.cos(np.deg2rad(lat[ilat])) *  v[0,ilat] * p[0]
        for ilev in range(p.shape[0])[1:]:
            psi[ilev,ilat] = psi[ilev-1,ilat] + coeff*np.cos(np.deg2rad(lat[ilat])) \
                             * v[ilev,ilat] * (p[ilev]-p[ilev-1])
    # Make into an xarray DataArray
    
    return psi

def v(ds, p, lat):
    ''' Take annual mean of meridional wind speed by taking a mean along time and longitude dimensions 
        Use this to calculate the streamfunction the dedicated function
    '''
    v_anm = ds.vcomp.mean(dim='time').mean(dim='lon').data
    psi = calc_streamfn(v_anm, p, lat)
    
    return psi

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
    
    return z_xr
       

def plots(ds, i):
    ''' Plots '''    
    lon = ds.coords['lon'].data
    lat = ds.coords['lat'].data
    p = ds.coords['pfull'].data
    z = altitude(p)
    
    temp = T(ds, z, lat)
    lvls1 = np.arange(82, 92, 1)
    u = uz(ds, z, lat)
    lvls2 = np.arange(-0.45, 0.5, 0.05)
    msf = v(ds, p, lat)
    msfz_xr = xr.DataArray(msf, coords=[z,lat], dims=['pfull','lat'])
    msfz_xr.attrs['units'] = 'kg/s'
    lvls3 = [-10e6, -5e6, -3e6, -1e6, -500e3, -300e3, -100e3, -50e3, -30e3, -10e3, -5e3, -3e3, -1e3, 0, 1e3, 3e3, 5e3, 10e3, 30e3, 50e3, 100e3, 300e3, 500e3, 1e6, 3e6, 5e6, 10e6]

    fig1 = plt.figure(figsize=(10,8))
    ax1 = fig1.add_subplot(111)
    cs1 = temp.plot.contourf(levels=lvls1)
#    fig1.colorbar(cs1, pad=0.03).set_label(label='[K]', size='xx-large')
    ax1.contour(cs1, colors='gainsboro', linewidths=1)
    #plt.clabel(cs1, inline=True, fontsize='large', colors='k')
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.title('%i'%(i), fontsize='xx-large', pad=10)
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    plt.savefig("TitanT_%i.png"%(i+1), bbox_inches = 'tight')
    plt.close()
    
    
    fig2 = plt.figure(figsize=(10,8))
    ax2 = fig2.add_subplot(111)
#    cs =uz(ds, p, lat).plot.contourf(levels=lvls1, cmap='RdBu_r', yincrease=False) #contour plot
#    ax1.set_ylim(max(p), 120)
#    ax1.set_ylabel('Pressure (hPa)')
#    ax1.set_yscale('log', basey=10) 
    cs2 = u.plot.contourf(cmap = 'RdBu_r', levels=lvls2) #contour plot
    ax2.contour(cs2, colors='gainsboro', linewidths=1)
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.title('%i'%(i+1), fontsize='xx-large', pad=10)
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    plt.savefig("Titanu_%i.png"%(i), bbox_inches = 'tight')
    plt.close()

    fig3 = plt.figure(figsize=(10,8))
    ax3 = fig3.add_subplot(111)
    cs3 = msfz_xr.plot.contourf(cmap = 'RdBu_r', levels=lvls3) #contour plot
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.title('%i'%(i+1), fontsize='xx-large', pad=10)
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    plt.title('%i'%(i+1))
    plt.savefig("Titanmsf_%i.png"%(i), bbox_inches = 'tight')
    plt.close()
       
    return plt.show()


if __name__ == '__main__': 
    files = sorted(glob.glob('grey_titan/year/run*/*.nc'))
    print(files)
    
    for i in np.arange(0, len(files)):
        file = files[i]
        ds = xr.open_dataset(file, decode_times=False)
        plots(ds, i)
    
    #Merge all plots into a GIF for visualisation
    images1 = glob.glob('Titanu*.png')
    list.sort(images1, key = lambda x: int(x.split('_')[1].split('.png')[0]))
    IMAGES1 = []
    for i in range(0,len(images1)):
        IMAGES1.append(imageio.imread(images1[i]))
    imageio.mimsave("Titan_u.gif", IMAGES1, 'GIF', duration = 1/2)
        
    #Delete all temporary plots from working directory
    for i in range(0,len(images1)):
        os.remove(images1[i])
        
    #Merge all plots into a GIF for visualisation
    images2 = glob.glob('Titanmsf*.png')
    list.sort(images2, key = lambda x: int(x.split('_')[1].split('.png')[0]))
    IMAGES2 = []
    for i in range(0,len(images2)):
        IMAGES2.append(imageio.imread(images2[i]))
    imageio.mimsave("Titan_msf.gif", IMAGES2, 'GIF', duration = 1/2)
        
    #Delete all temporary plots from working directory
    for i in range(0,len(images2)):
        os.remove(images2[i])
        
    #Merge all plots into a GIF for visualisation
    images3 = glob.glob('TitanT_*.png')
    list.sort(images3, key = lambda x: int(x.split('_')[1].split('.png')[0]))
    IMAGES3 = []
    for i in range(0,len(images3)):
        IMAGES3.append(imageio.imread(images3[i]))
    imageio.mimsave("Titan_T.gif", IMAGES3, 'GIF', duration = 1/2)
        
    #Delete all temporary plots from working directory
    for i in range(0,len(images3)):
        os.remove(images3[i])


