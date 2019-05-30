"""
    This script plots 4 different temperature profiles against the huygens data.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob
import xlrd

def uz(ds, z):
    ''' Take mean of zonal wind speed by taking a mean along time and longitude dimensions '''
    u_anm = ds.ucomp.mean(dim='time')
    u_avg = u_anm.mean(dim='lat').mean(dim='lon') #global mean
    u_huygens = u_anm[:,28, 60] #around the location of the Huygens site
    
    return u_huygens

def T(ds, z):
    ''' Take mean of average zonal temperature by taking averages along time and longitude dimensions '''
    T_anm = ds.temp.mean(dim='time')
    T_huygens = T_anm[:,28, 60].data #around the location of the Huygens site
    T_avg = T_anm.mean(dim='lat').mean(dim='lon') #global mean
    
    return T_huygens

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
        vel.append(sheet.cell_value(i,1)) #/5)
    vel = [float(x) for x in vel]
    alt = [float(x) for x in alt]
    
    return vel, alt


def plots(p, z, files, colors, label):
    """Plots the profiles of the 2 variables on separate figures"""
    
    u_huy, z_huy1 = Huygensu()
    T_huy, z_huy2 = HuygensT()
    
    fig1, ax1 = plt.subplots(figsize=(7.5, 6.5))
    ax1.set_xlabel('Zonal Wind (m s$^{-1}$)', fontsize='xx-large')
    
    for i in range(len(files)):
        ax1.plot(uz(files[i], z), z, color=colors[i], linewidth=3)
             
    ax1.plot(u_huy, z_huy1, color='k', linewidth=2)
    ax1.xaxis.set_tick_params(labelsize = 'xx-large')
    ax1.yaxis.set_tick_params(labelsize = 'xx-large')
    ax1.set_ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    ax1.set_ylim(min(z), 50)
    ax1.set_xlim(-3, 5)
    plt.legend(labels=(label[0], label[1], label[2], label[3], 'Huygens'), loc=0, #upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=False, shadow=False, ncol=1, fontsize='x-large')
    plt.title('(a)', fontsize='xx-large', pad=10)
    plt.tight_layout()
    
    fig2 = plt.figure(figsize=(7.5,6.5))
    ax2 = fig2.add_subplot(111)   
    ax2.set_xlabel('Temperature (K)', fontsize='xx-large')
    
    for j in range(len(files)):
        ax2.plot(T(files[j], z), z, color=colors[j], linewidth=3)
        
    ax2.plot(T_huy, z_huy2, color='k', linewidth=2)
    ax2.set_ylabel('Pseudo-altitude (km)', fontsize='xx-large')
    ax2.xaxis.set_tick_params(labelsize = 'xx-large')
    ax2.yaxis.set_tick_params(labelsize = 'xx-large')
    ax2.set_ylim(min(z), 50)
    ax2.set_xlim(68, 157)
#    plt.plot([min(T_huy)-1, 170], [44,44], 'k--')
#    plt.text(114, 46, 'Tropopause', fontsize='xx-large')
#    plt.text(130, 50, 'Stratosphere', fontsize='xx-large')
    plt.legend(labels=(label[0], label[1], label[2], label[3], 'Huygens'),loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=False, shadow=True, ncol=5, fontsize='x-large')
    plt.title('(b)', fontsize='xx-large', pad=10)
    plt.tight_layout()
    
    return plt.show()

if __name__ == '__main__':   
    files1 = sorted(glob.glob('grey_titan1/winter/run*/*.nc'))
    ds1 = xr.open_mfdataset(files1, decode_times=False) #multiple files = mf, data for a year (12 months)
    label1 = 'B'
    
    files2 = sorted(glob.glob('grey_titan2/winter/run*/*.nc'))
    ds2 = xr.open_mfdataset(files2, decode_times=False) #multiple files = mf, data for a year (12 months)
    label2 = 'C'
    
    files3 = sorted(glob.glob('grey_titan3/winter/run*/*.nc'))
    ds3 = xr.open_mfdataset(files3, decode_times=False) #multiple files = mf, data for a year (12 months)
    label3 = 'D'
        
    files4 = sorted(glob.glob('grey_titan4/winter/run*/*.nc'))
    ds4 = xr.open_mfdataset(files4, decode_times=False) #multiple files = mf, data for a year (12 months)
    label4 = 'G'
    
    file_list = [ds1, ds2, ds3, ds4]
    label_list = [label1, label2, label3, label4]
    color_list = ['#2980B9', '#27AE60', '#9B59B6', '#C0392B']
    
    p = ds1.coords['pfull'].data
    z = altitude(p)
    
    plots(p, z, file_list, color_list, label_list)
