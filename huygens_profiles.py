"""
    This script plots the Huygens data for temperature and zonal wind.
    Download from the ESA Public Archive.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import xlrd

def altitude(p):
    """Finds altitude from pressure using z = -H*log10(p/p0) """
    
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

def pressure(z):
    """Finds pressure from altitude using z = -H*log10(p/p0) """
    
    R = 290 #specific gas constant 
    T = 93.65 #surface temperature K from A.Coustenis book
    g = 1.354 #surface gravity from A.Coustenis book
    p0 = 1467 #surface pressure in hPa
    
    p = np.empty_like(z)
    
    for i in range(p.shape[0]):
        p[i] = p0*np.exp((-1)*z[i]*(10**3)/((R*T/g)))
        
    # Make into an xarray DataArray
    p_xr = xr.DataArray(p, coords=[p], dims=['pfull'])
    p_xr.attrs['units'] = 'hPa'
    
    return p_xr


def HuygensT():
    """Temperature profile"""
    loc =('T_huygens.xlsx')
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    alt = []
    temp= []

    for i in range(sheet.nrows):
        alt.append(sheet.cell_value(i,1))
        temp.append(sheet.cell_value(i,2))
    temp = [float(x) for x in temp]
    alt = [float(x) for x in alt]
    
    return temp, alt

def Huygensu():
    """Zonal wind profile"""
    
    loc =('u_huygens.xlsx')
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    alt = []
    vel = []

    for i in range(sheet.nrows):
        alt.append(sheet.cell_value(i,0))
        vel.append(sheet.cell_value(i,1))
    vel = [float(x) for x in vel]
    alt = [float(x) for x in alt]
    
    return vel, alt


def plots():
    """Plots the 2 profiles on separate figures"""
    
    v_huy, z_huy1 = Huygensu()
    T_huy, z_huy2 = HuygensT()
    p_huy1 = pressure(z_huy1)    
    p_huy2 = pressure(z_huy2)
    
    fig1, ax1 = plt.subplots(figsize=(7.5,6.5))
    ax3 = ax1.twinx()
    ax1.set_xlabel('Zonal Wind (m/s)', fontsize='xx-large')
    ax3.plot(v_huy, z_huy1, color='k', linewidth=2.5)
    ax1.plot(v_huy, p_huy1, color='k', linewidth=2.5)
    ax1.set_ylim(max(p_huy1), 10.028459244111575)
    ax1.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    ax1.set_yscale('log', basey=10) 
    ax3.set_ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    ax3.set_ylim(min(z_huy1), 100)
    ax3.set_xlim(-10,60)
    plt.plot([-10, 70], [44,44], 'k--')
    plt.text(40, 46, 'Tropopause', fontsize='xx-large')
    ax1.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    ax3.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.title('(b)', fontsize='xx-large', pad='20')
    plt.tight_layout()
    
    fig2 = plt.figure(figsize=(7.5,6.5))
    ax2 = fig2.add_subplot(111) 
    ax4 = ax2.twinx()
    ax2.set_xlabel('Temperature (K)', fontsize='xx-large')   
    ax4.plot(T_huy, z_huy2, color='k', linewidth=2.5)
    ax2.plot(T_huy, p_huy2, color='k', linewidth=2.5)
    ax2.set_ylim(max(p_huy2), 10.028459244111575)
    ax2.set_ylabel('Pressure (hPa)', fontsize='xx-large')
    ax2.set_yscale('log', basey=10) 
    ax4.set_ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    ax4.set_ylim(min(z_huy1), 100)
    ax4.set_xlim(70, 145)
    plt.plot([70, 145], [44,44], 'k--')
    plt.text(123, 46, 'Tropopause', fontsize='xx-large')
#    plt.text(120, 60, 'Stratosphere', fontsize='xx-large')
    ax2.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    ax4.tick_params(axis='both', labelsize = 'xx-large', direction='in', which='both')
    plt.title('(a)', fontsize='xx-large', pad='20')
    plt.tight_layout()
    
    return plt.show()

if __name__ == '__main__': 
    plots()
