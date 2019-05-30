"""
    This script plots differences between 2 datasets.
    Important that they are of the same resolution (e.g. both T21 or both T42)
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
    T_anm = ds.t_surf.mean(dim='time')

    return T_anm

def T(ds, z, lat):
    ''' Take mean of average zonal temperature by taking averages along time and longitude dimensions '''
    T_anm = ds.temp.mean(dim='time').mean(dim='lon').data 
    
    return T_anm

def P_surf(ds, lat, lon):
    ''' Take mean of surface temperature by taking a mean along time dimension '''
    p_anm = ds.ps.mean(dim='time').data 
    
    return p_anm

def uz(ds, z, lat):
    ''' Take mean of zonal wind speed by taking a mean along time and longitude dimensions '''
    u = ds.ucomp
    u_anm = u.mean(dim='time').mean(dim='lon').data 
    
    return u_anm

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
    v_anm = ds.vcomp.mean(dim='time').mean(dim='lon')
    psi = calc_streamfn(v_anm, p, lat)
    
    return psi

def altitude(p):
    """Finds altitude from pressure using z = -H*log10(p/p0) """
    
    R = 290 #specific gas constant 
    T = 93.65 #surface temperature K from A.Coustenis book
    g = 1.354 #surface gravity from A.Coustenis book
    p0 = 6.1 #surface pressure in hPa
    
    z = np.empty_like(p)
    
    for i in range(p.shape[0]):
        z[i] = (-1)*(R*T/g)*np.log((p[i])/p0)/(10**3)
        
    # Make into an xarray DataArray
    z_xr = xr.DataArray(z, coords=[z], dims=['pfull'])
    z_xr.attrs['units'] = 'km'
    
    #below is the inverse of the calculation
    #p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
    
    return z_xr

def difference(a1, a2, coord1, coord2, dim1, dim2, unit):
    """Make the difference between 2 datasets an xarray"""
    
    diff = a1 - a2
    
    diff_xr = xr.DataArray(diff, coords=[coord1, coord2], dims=[dim1, dim2])
    diff_xr.attrs['units'] = unit
    
    return diff_xr

def diff_variables(ds1, ds2, lat, lon, z, p):
    """Find difference between datasets"""
    
    Tsurf1 = ds1.t_surf.mean(dim='time')
    Tsurf2 = ds2.t_surf.mean(dim='time')
    Tsurf_diff = difference(Tsurf1, Tsurf2, lat, lon, 'lon', 'lat', 'K')
    
    Psurf1 = ds1.ps.mean(dim='time')
    Psurf2 = ds2.ps.mean(dim='time')
    Psurf_diff = difference(Psurf1, Psurf2, lat, lon, 'lon', 'lat', 'Pa')
    
    T1 = ds1.temp.mean(dim='time').mean(dim='lon')
    T2 = ds2.temp.mean(dim='time').mean(dim='lon')
    T_diff = difference(T1, T2, z, lat, 'lat', 'pfull', 'K')
    
    uz1 = ds1.ucomp.mean(dim='time').mean(dim='lon')
    uz2 = ds2.ucomp.mean(dim='time').mean(dim='lon')
    uz_diff = difference(uz1, uz2, z, lat, 'lat', 'pfull', 'm/s')
    
    msf1 = v(ds1, p, lat)
    msf2 = v(ds2, p, lat)
    msf_diff = difference(msf1, msf2, z, lat, 'lat', 'pfull', 'kg/s')
    
    return Tsurf_diff, Psurf_diff, T_diff, uz_diff, msf_diff


def plots(ds1, ds2):
    ''' Plots '''    
    lat = ds1.coords['lat'].data
    lon = ds1.coords['lon'].data
    p = ds1.coords['pfull'].data
    z = altitude(p)
    
    Tsurf, Psurf, Temp, Speed, MSF = diff_variables(ds1, ds2, lat, lon, z, p)

    spacing=20
    majorLocator=MultipleLocator(spacing)
    
    fig1 = plt.figure(figsize=(6,4))
    ax1 = fig1.add_subplot(111)
    cs1 = Tsurf.plot.contourf(levels=15) #contour plot
    ax1.contour(cs1, colors='gainsboro', linewidths=0.5)
    #plt.colorbar(labelsize='xx-large')
    plt.xlabel('Longitude ('+chr(176)+')', fontsize='xx-large')    
    plt.xticks(np.arange(0, 420, 60))
    plt.yticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.ylabel('Latitude', fontsize='xx-large')
    plt.title('(c) $T_{0}$ Difference', fontsize='xx-large', pad=10)
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
#    cs = Temp.plot.contourf(levels=np.arange(70, 90, 0.5), yincrease=False) #contour plot
#    ax3.set_ylim(max(p), 120)
#    ax3.set_ylabel('Pressure (hPa)')
#    ax3.set_yscale('log', basey=10) 
    cs = Temp.plot.contourf(levels=51, yincrease=True) #contour plot
    plt.ylabel('Pseudo-Altitude (km)', fontsize='large')
    plt.ylim(min(z), 50)
    ax3.contour(cs, colors='gainsboro', linewidths=1)
    ax3.xaxis.set_tick_params(labelsize = 'large')
    ax3.yaxis.set_tick_params(labelsize = 'large')    
    plt.title('Titan Mean Zonal Temperature')
    plt.xlabel('Latitude (degrees)', fontsize='large')   
    
    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    cs2 = Speed.plot.contourf(levels=np.arange(-5, 5, 0.5), cmap='RdBu_r')
    ax2.contour(cs2, colors='gainsboro', linewidths=1)
#    fig2.colorbar(cs2, pad=0.03).set_label(label='[$m s^{-1}$]', size='xx-large')
    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    plt.ylim(min(z), 50)
    plt.yticks([10, 20, 30, 40, 50])
    plt.yticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
    plt.xlabel('Latitude', fontsize='xx-large')
    plt.title('(b) Zonal Wind', fontsize='xx-large', pad=10)
    ax2.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.tight_layout()
    
#    lvls = [-10e6, -5e6, -3e6, -1e6, -500e3, -300e3, -100e3, -50e3, -30e3, -10e3, -5e3, -3e3, -1e3, 0, 1e3, 3e3, 5e3, 10e3, 30e3, 50e3, 100e3, 300e3, 500e3, 1e6, 3e6, 5e6, 10e6]
#
#    
#    fig3 = plt.figure(figsize=(6.15,4))
#    ax3 = fig3.add_subplot(111)
#    cs3 = MSF.plot.contourf(levels=lvls, cmap='RdBu_r')
##    fig3.colorbar(cs3, pad=0.03).set_label(label='[$kg s^{-1}$]', size='xx-large')
#    plt.ylabel('Pseudo-altitude (km)', fontsize='xx-large')
#    plt.ylim(min(z), 50)
#    plt.yticks([-90, -45, 0, 45, 90], ['90S', '45S', '0', '45N', '90N'])
#    plt.xlabel('Latitude', fontsize='xx-large')
#    plt.title('(c) Streamfunction', fontsize='xx-large', pad=10)
#    ax3.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
#    plt.tight_layout()

    return plt.show()


if __name__ == '__main__': 
    files1 = sorted(glob.glob('grey_titan1/winter/run*/*.nc'))
    ds1 = xr.open_mfdataset(files1, decode_times=False) #multiple files = mf, data for a year (12 months)
    
    files2 = sorted(glob.glob('grey_titan2/winter/run*/*.nc'))
    ds2 = xr.open_mfdataset(files2, decode_times=False) #multiple files = mf, data for a year (12 months)
    
    plots(ds1, ds2)
