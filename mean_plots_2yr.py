"""
    This script plots seasonal averages for 2 years. Can be expanded to more.
    Sorry it's a bit of a messy script.
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
import xarray as xr
import numpy as np
import glob

def T_surf(ds, lat, lon):
    ''' Take mean of surface temperature by taking a mean along time dimension '''
    T_anm = ds.t_surf.mean(dim='time').data 
    T_xr = xr.DataArray(T_anm, coords=[lat,lon], dims=['lat','lon']) #xarray DataArray
    T_xr.attrs['units']='K'
    
    return T_xr

def sw_toa(ds, lat, lon):
    ''' Take mean of incoming sw by taking a mean along time dimension '''
    sw_anm = ds.swdn_toa.mean(dim='time').data 
    sw_xr = xr.DataArray(sw_anm, coords=[lat,lon], dims=['lat','lon']) #xarray DataArray
    sw_xr.attrs['units']='W/m2'
    
    return sw_xr

def P_surf(ds, lat, lon):
    ''' Take mean of surface temperature by taking a mean along time dimension '''
    p_anm = ds.ps.mean(dim='time').data 
    p_xr = xr.DataArray(p_anm, coords=[lat,lon], dims=['lat','lon']) #xarray DataArray
    p_xr.attrs['units']='Pa'
    
    return p_xr

def T(ds, z, lat):
    ''' Take mean of average zonal temperature by taking averages along time and longitude dimensions '''
    T_anm = ds.temp.mean(dim='time').mean(dim='lon').data 
    #ds.temp.mean(dim='time').mean(dim='lon')[13]
    T_xr = xr.DataArray(T_anm, coords=[z,lat], dims=['pfull','lat']) #xarray DataArray
    T_xr.attrs['units']='K'
    
    return T_xr

def uz(ds, z, lat):
    ''' Take mean of zonal wind speed by taking a mean along time and longitude dimensions '''
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
    """Finds altitude from pressure using z = -H*log10(p/p0) """
    
    R = 290 #specific gas constant 
    T = 93.65 #surface temperature K from A.Coustenis book
    g = 1.354 #surface gravity from A.Coustenis book
    p0 = 1467 #surface pressure in hPa 6.1 for mars
    
    z = np.empty_like(p)
    
    for i in range(p.shape[0]):
        z[i] = (-1)*(R*T/g)*np.log((p[i])/p0)/(10**3)
        
    # Make into an xarray DataArray
    z_xr = xr.DataArray(z, coords=[z], dims=['pfull'])
    z_xr.attrs['units'] = 'km'
    
    #below is the inverse of the calculation
    #p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
    
    return z_xr

def average(a1, a2, coord1, coord2, dim1, dim2, unit):
    """Make the average an xarray"""
    
    avg = (a1 + a2)/2
    
    avg_xr = xr.DataArray(avg, coords=[coord1, coord2], dims=[dim1, dim2])
    avg_xr.attrs['units'] = unit
    
    return avg_xr

def avg_variables(ds1, ds2, lat, lon, z, p):
    """Find average of 2 periods"""
    
    T1 = ds1.temp.mean(dim='time').mean(dim='lon')
    T2 = ds2.temp.mean(dim='time').mean(dim='lon')
    T_avg = average(T1, T2, z, lat, 'lat', 'pfull', 'K')
    
    uz1 = ds1.ucomp.mean(dim='time').mean(dim='lon')
    uz2 = ds2.ucomp.mean(dim='time').mean(dim='lon')
    uz_avg = average(uz1, uz2, z, lat, 'lat', 'pfull', 'm/s')
    
    msf1 = v(ds1, p, lat)
    msf2 = v(ds2, p, lat)
    msf_avg = average(msf1, msf2, z, lat, 'lat', 'pfull', 'kg/s')
    
    return T_avg, uz_avg, msf_avg

def plots(ds1, ds2):
    ''' Plots '''    
    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    z = altitude(p)

    temp, u, msfz_xr = avg_variables(ds1, ds2, lat, lon, z, p)
    lvls1 = np.arange(80, 86, 0.5)
    lvls2 = np.arange(-0.4, 0.45, 0.05)
    lvls3 = [-10e6, -5e6, -3e6, -1e6, -500e3, -300e3, -100e3, -50e3, -30e3, -10e3, -5e3, -3e3, -1e3, 0, 1e3, 3e3, 5e3, 10e3, 30e3, 50e3, 100e3, 300e3, 500e3, 1e6, 3e6, 5e6, 10e6]

#    fig1 = plt.figure(figsize=(6,4))
#    ax1 = fig1.add_subplot(111)
#    cs1 = T_surf(ds, lat, lon).plot.contourf(cmap = 'RdBu_r', levels=11) #contour plot
#    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
#    #plt.colorbar(labelsize='xx-large')
#    plt.xlabel('Longitude'+chr(176), fontsize='xx-large')    
#    plt.xticks(np.arange(0, 420, 60))
#    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
#    plt.ylabel('Latitude', fontsize='xx-large')
#    plt.title('(?) Surface Temperature', fontsize='xx-large', pad=10)
#    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
#    plt.tight_layout()


    fig1 = plt.figure(figsize=(6,4))
    ax1 = fig1.add_subplot(111)
    cs1 = temp.plot.contourf(levels=lvls1)
#    fig1.colorbar(cs1, pad=0.03).set_label(label='[K]', size='xx-large')
    ax1.contour(cs1, colors='gainsboro', linewidths=1)
    #plt.clabel(cs1, inline=True, fontsize='large', colors='k')
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 10)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.title('(a) Temperature', fontsize='xx-large', pad=10)
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    
    
    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    cs2 = u.plot.contourf(levels=lvls2, cmap='RdBu_r')
    ax2.contour(cs2, colors='gainsboro', linewidths=1)
#    fig2.colorbar(cs2, pad=0.03).set_label(label='[$m s^{-1}$]', size='xx-large')
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.title('(b) Zonal Wind', fontsize='xx-large', pad=10)
    ax2.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
   
    fig3 = plt.figure(figsize=(6.15,4))
    ax3 = fig3.add_subplot(111)
    cs3 = msfz_xr.plot.contourf(levels=lvls3, cmap='RdBu_r')
#    fig3.colorbar(cs3, pad=0.03).set_label(label='[$kg s^{-1}$]', size='xx-large')
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.xticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.title('(c) Streamfunction', fontsize='xx-large', pad=10)
    ax3.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    
    return plt.show()



if __name__ == '__main__': 
    files1 = sorted(glob.glob('grey_titan/winter1/run*/*.nc'))
    ds1 = xr.open_mfdataset(files1, decode_times=False) #multiple files = mf, data for a year (12 months)
    
    files2 = sorted(glob.glob('grey_titan/winter2/run*/*.nc'))
    ds2 = xr.open_mfdataset(files2, decode_times=False) #multiple files = mf, data for a year (12 months)

    
    plots(ds1, ds2)