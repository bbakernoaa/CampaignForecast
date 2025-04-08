!/usr/bin/env python
# coding: utf-8

# In[1]:

# =======================================================================================
# This is the forecast template originally deveopped for AEROMMA and USOS field campaigns
# A number of products are obtained from various sources. 
# Processed plots/animations are eventually packed into a PPT file.
# Contact: Siyuan Wang (siyuan.wang@noaa.gov)
# =======================================================================================
import pandas as pd
import numpy as np
import xarray as xr

import aiobotocore
import s3fs

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import rc
from matplotlib.image import imread
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.patches as mpatches
from datetime import date, datetime, timedelta
import time,os,gc,warnings
import pytz, timezonefinder

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
# ### cartopy.config['data_dir'] = '/home/swang/.local/share/cartopy/shapefiles/natural_earth/cultural'
# cartopy.config['data_dir'] = '/home/swang/.local/share/cartopy/shapefiles/natural_earth/cultural'
# cartopy.config['data_dir'] = '/home/swang/.local/share/cartopy/'
# cartopy.config['data_dir'] = '/home/swang/.local/share/cartopy/shapefiles'
# cartopy.config['data_dir'] = '/home/swang/.local/share/cartopy/shapefiles/natural_earth'

import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint,potential_temperature

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

import requests
from bs4 import BeautifulSoup
import urllib.request
from PIL import Image
import imageio
import sys
import ftplib

import os
import openai

plt.style.use('bmh')


# In[2]:


import dask
import dask.dataframe as dd
import imageio.v3 as iio
from xml.etree import ElementTree as ET
from dateutil.parser import parse

# standard2.py and epa_util.py are also needed but will be provided as python scripts
from standard2 import * 


# In[3]:


# ===========
# FPT related
# ===========
def upload_to_FTP(in_local_fname, in_remote_path, in_remote_fname, in_ftp_server, in_ftp_username, in_ftp_password):
    try:
        session = ftplib.FTP(in_ftp_server, in_ftp_username, in_ftp_password)
        session.set_pasv(True)
        file2upload = open(in_local_fname,'rb')
        session.storbinary(f"STOR /{in_remote_path}/{in_remote_fname}", file2upload)
    except:
        print('*** FTP upload error ***')


def if_dir_on_FTP(in_server, in_username, in_password, in_dir):
    def directory_exists(ftp, in_directory):
        file_list = []
        ftp.retrlines('LIST', file_list.append)
        for line in file_list:
            if line.split()[-1] == in_directory and line.upper().startswith('D'):
                return True
        return False
    try:
        with ftplib.FTP(in_server, in_username, in_password) as session:
            dir_list = in_dir.split('/')
            for sub_dir in dir_list:
                if not directory_exists(session, sub_dir):
                    session.cwd(sub_dir)
                else:
                    session.cwd(sub_dir)
            current_directory = session.pwd()
            if current_directory == '/' + in_dir:
                # print(f"YES, '{in_dir}' does exist on FTP.")
                return True
            else:
                # print(f"No, '{in_dir}' does not exist on FTP.")
                return False
    except Exception as e:
        # print(f"Error checking {str(e)} on FTP")
        return False
        

def create_dir_on_FTP(in_dir, in_ftp_server, in_ftp_username, in_ftp_password):
    session = ftplib.FTP(in_ftp_server, in_ftp_username, in_ftp_password)
    try: session.mkd(in_dir)
    except Exception as e: print(f"Error creating '{in_dir}' on FTP: {e}")


# In[4]:


DataDump = '/home/swang/DataDump/'
OutputDir = '/home/swang/AEROMMA_tmp/'

timer_start = time.time()

# --- get GOES animation for yesterday
prod_goes_yyyymmdd = str(datetime.today()-timedelta(days=1)).split(' ')[0].replace('-','')

# --- get HRRR forecast for today
hrrr_init_yyyymmdd = str(datetime.today()).split(' ')[0].replace('-','')

# --- setup FTP
ftp_server = '***'
ftp_username = '***'
ftp_password = '***'

ftp_dir_daily = f'AutoPresentations/tmp_{hrrr_init_yyyymmdd}'
print(f'remote daily dir on FTP: {ftp_dir_daily}')

if if_dir_on_FTP(ftp_server, ftp_username, ftp_password, ftp_dir_daily)==False:
    create_dir_on_FTP(ftp_dir_daily, ftp_server, ftp_username, ftp_password)


# In[5]:


in_config_loc = sys.argv[1]

# in_config_loc = 'SLC'
# in_config_loc = 'DEN'
# in_config_loc = 'Chicago'

config_fname = '/home/swang/AEROMMA_scripts/config_%s' % in_config_loc

f_config = open(config_fname, 'r')
f_config_contents = f_config.readlines()
f_config.close()

for line in f_config_contents:
    if line.replace(' ','')[0]!='#':
        line = line.split('#')[0].replace('\n','')
        # print(line)
        if ('prod_loc' in line) & ('prod_lat' in line) & ('prod_lon ' in line):
            prod_loc = line.split('=')[-1].split(',')[0].replace("'","").replace(" ","")
            prod_lat = float(line.split('=')[-1].split(',')[1])
            prod_lon = float(line.split('=')[-1].split(',')[2])
        if 'NWS_site_id' in line:
            NWS_site_id = line.split('=')[1].replace("'","").replace(" ","")
        if 'prod_goes_start_hhmmss' in line:
            prod_goes_start_hhmmss = str(line.split('=')[1].replace(' ','').replace("'",""))
        if 'which_goes' in line:
            which_goes = line.split('=')[1].replace(' ','').replace("'","")
        if 'airports_of_interest' in line:
            for s in [" ","[","]","'"]: line = line.replace(s,'')
            airports_of_interest = line.split('=')[1].split(',')
        if 'hrrr_init_hr' in line:
            hrrr_init_hr = int(line.split('=')[1])
        if 'hrrr_fcst_hr_list' in line:
            for s in [" ","[","]"]: line = line.replace(s,'')
            hrrr_fcst_hr_list = [int(s) for s in line.split('=')[1].split(',')]

print('\n--- done loading config file:')
print(prod_loc, prod_lat, prod_lon)
print(NWS_site_id)
print(prod_goes_start_hhmmss)
print(which_goes)
print(airports_of_interest)
print(hrrr_init_hr)
print(hrrr_fcst_hr_list)


# In[6]:


# # --- set location
# prod_loc, prod_lat, prod_lon = 'NYC', 40.7128, -74.0060
# # prod_loc, prod_lat, prod_lon = 'LA', 34.0522, -118.2437
# # prod_loc, prod_lat, prod_lon = 'Chicago', 41.8781, -87.6298

# # --- NWS forecast discussion link
# #     NYC: https://forecast.weather.gov/product.php?site=OKX&issuedby=OKX&product=AFD&format=txt&version=1&glossary=0
# #     LA: https://forecast.weather.gov/product.php?site=LOX&issuedby=LOX&product=AFD&format=TXT&version=1&glossary=0
# #     Chicago: https://forecast.weather.gov/product.php?site=LOT&issuedby=LOT&product=AFD&format=TXT&version=1&glossary=0
# NWS_site_id = 'OKX'  # NYC
# # NWS_site_id = 'LOX'  # LA
# # NWS_site_id = 'LOT'  # Chicago

# # --- GOES images: previous day, morning to afternoon
# prod_goes_start_hhmmss = '140000'
# which_goes = 'goes16'

# # --- ACARS sounding setup: airport(s)
# airports_of_interest = ['JFK','LGA','EWR']

# # --- HRRR forecast for today
# hrrr_init_hr = 10    # this is 6AM EDT
# hrrr_fcst_hr_list = [3,4,5,6,7,8,9,10]   # this is from 9AM to 6PM EDT


# In[ ]:





# In[7]:


# ==============
# GOES functions
# ==============
session = aiobotocore.session.AioSession()
fs = s3fs.S3FileSystem(anon=True,session=session)  # no need to/don't run this shit for each fucking file

def get_goes_datetime(in_prod,in_yymmdd,in_hhmmss,in_goes='goes17'):
#     session = aiobotocore.session.AioSession()
#     fs = s3fs.S3FileSystem(anon=True,session=session)
    y,m,d,h = int(in_yymmdd[0:4]),int(in_yymmdd[4:6]),int(in_yymmdd[6:8]),int(in_hhmmss[:2])
    # doy = (date(y,m,d)-date(y, 1, 1)).days+1
    doy = (datetime(y,m,d)-datetime(y, 1, 1)).days+1
    awsdir = '/%04d/%02d/%02d' % (y,doy,h)
    flist = fs.ls('noaa-'+in_goes+'/'+in_prod+awsdir)
    time_flist = np.array([int(f.split('/')[-1].split('_')[-2].replace('e','')[7:13]) for f in flist])
    if len(time_flist)>0:
        f = flist[ abs(time_flist-int(in_hhmmss)).argmin() ]
        fname = f.split('/')[-1]
        # print(f)
        # print(fname)
        # --- downloads the file to the local dir
        if (os.path.exists(DataDump+fname)==False): fs.get(f, DataDump+fname)
        ds_mcmipc = xr.open_dataset(DataDump+fname)
        # --- add lat lon to it
        ds_mcmipc = add_lat_lon_to_goes(ds_mcmipc)
        # --- Height of the satellite's orbit
        sat_h = ds_mcmipc.goes_imager_projection.perspective_point_height
        # --- Longitude of the satellite's orbit
        sat_lon = ds_mcmipc.goes_imager_projection.longitude_of_projection_origin
        X = ds_mcmipc.variables['x'][:] * sat_h
        Y = ds_mcmipc.variables['y'][:] * sat_h
        globe = ccrs.Globe(semimajor_axis=ds_mcmipc.goes_imager_projection.semi_major_axis, 
                           semiminor_axis=ds_mcmipc.goes_imager_projection.semi_minor_axis)
        img_proj = ccrs.Geostationary(satellite_height=sat_h,central_longitude=sat_lon,globe=globe)
        img_extent = (X.values.min(), X.values.max(), Y.values.min(), Y.values.max())
        if in_prod=='ABI-L2-MCMIPC':
            # --- get geo color based on red, blue, and veggie channels in GOES
            ds_geocolor = get_geocolor(ds_mcmipc)    
            return ds_mcmipc,ds_geocolor,img_proj,img_extent,fname
        else:
            return ds_mcmipc,'fuck',img_proj,img_extent,fname        
    else:
        return 'fuck','fuck','fuck','fuck','fuck'
    

def add_lat_lon_to_goes(ds):
    warnings.filterwarnings("ignore")
    x, y = ds.x, ds.y
    goes_imager_projection = ds.goes_imager_projection
    x,y = np.meshgrid(x,y)
    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi/180.)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat
    a = np.sin(x)**2 + (np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2))
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2
    r_s = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)
    lat = np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((H-s_x)**2 +s_y**2))) * (180/np.pi)
    lon = (l_0 - np.arctan(s_y / (H-s_x))) * (180/np.pi)
    ds = ds.assign_coords({"lat":(["y","x"],lat),
                           "lon":(["y","x"],lon)})
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    return ds


def contrast_correction(color, contrast):
    """
    Modify the contrast of an RGB
    See:
    https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/

    Input:
        color    - an array representing the R, G, and/or B channel
        contrast - contrast correction level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(color-.5)+.5
    COLOR = np.clip(COLOR, 0, 1)  # Force value limits 0 through 1.
    return COLOR


def get_geocolor(ds_mcmipc):
    # --- clip, then do a simple gamma correction
    vgamma = 2.2
    cmi_1 = np.clip(ds_mcmipc['CMI_C01'],0,1)**(1./vgamma)  # Blue visible
    cmi_2 = np.clip(ds_mcmipc['CMI_C02'],0,1)**(1./vgamma)  # Red visible
    cmi_3 = np.clip(ds_mcmipc['CMI_C03'],0,1)**(1./vgamma)  # Veggie near IR
    # --- Rebin function from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    def rebin(a, shape):
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.values.reshape(sh).mean(-1).mean(1)
    cmi_2_rebin = rebin(cmi_2, [1500, 2500])
    # cmi_2_rebin = rebin(cmi_2, [500, 500])
    # --- get green based on red, blue, and veggie
    # https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html
    # ref_gamma_true_green = 0.45 * cmi_2_rebin + 0.45 * cmi_1 + 0.1 * cmi_3
    ref_gamma_true_green = 0.48358168 * cmi_2_rebin + 0.45706946 * cmi_1 + 0.06038137 * cmi_3
    # --- stack RGB and get geo color
    geocolor = np.stack([cmi_2_rebin, ref_gamma_true_green, cmi_1], axis=2)
    # --- correct for contrast, make the img sharper
    geocolor = contrast_correction(geocolor, 105)
    return geocolor

def get_date_from_goes_file(in_filename):
    s,e = in_filename.split('_')[-3],in_filename.split('_')[-2]
    dt = [datetime.strptime(t[1:5]+'-'+t[5:8],"%Y-%j").strftime("%Y-%m-%d") + ' %s:%s:%s' % (t[8:10],t[10:12],t[12:14]) for t in [s,e]]
    return dt


def get_abi_frp_in_area_of_interest(in_area_of_interest,in_goes_fdcc_list):
    fdcc_frp_lat,fdcc_frp_lon,fdcc_frp,fdcc_frp_timestamp,fdcc_frp_mask,fdcc_frp_dqf = [],[],[],[],[],[]
    fdcc_frp_aggregrate,fdcc_frp_err_aggregrate,fdcc_frp_timestamp_aggregrate = [],[],[]
    for f in in_goes_fdcc_list:
        fname = f.split('/')[-1]
        fdcc = xr.open_dataset(f)
        fdcc = add_lat_lon_to_goes(fdcc)
        # --- if filter with Mask == 10 (good pixel) 13 (high probability fire pixel) then nothing left
        mask = (fdcc['lon']>=in_area_of_interest[0]) & (fdcc['lon']<=in_area_of_interest[1]) \
             & (fdcc['lat']>=in_area_of_interest[2]) & (fdcc['lat']<=in_area_of_interest[3]) \
             & (fdcc['DQF']<=1) ### & (fdcc['Mask']==30)

        fdcc_frp_lon.extend(fdcc['lon'].values.ravel()[mask.values.ravel()])
        fdcc_frp_lat.extend(fdcc['lat'].values.ravel()[mask.values.ravel()])
        fdcc_frp.extend(fdcc['Power'].values.ravel()[mask.values.ravel()])
        fdcc_frp_mask.extend(fdcc['Mask'].values.ravel()[mask.values.ravel()])
        fdcc_frp_dqf.extend(fdcc['DQF'].values.ravel()[mask.values.ravel()])
        fdcc.close()

        dt = get_date_from_goes_file(fname)[1]
        y,m,d = dt.split(' ')[0].split('-')[0],dt.split(' ')[0].split('-')[1],dt.split(' ')[0].split('-')[2]
        hh,mm,ss = dt.split(' ')[1].split(':')[0],dt.split(' ')[1].split(':')[1],dt.split(' ')[1].split(':')[2]
        dt = datetime(int(y),int(m),int(d),int(hh),int(mm),int(ss))

        fdcc_frp_timestamp.extend( [dt for v in fdcc['Power'].values.ravel()[mask.values.ravel()]] )

        fdcc_frp_aggregrate.append(np.nanmean(fdcc['Power'].values.ravel()[mask.values.ravel()]))
        fdcc_frp_err_aggregrate.append(np.nanstd(fdcc['Power'].values.ravel()[mask.values.ravel()]))
        fdcc_frp_timestamp_aggregrate.append(dt)

        print(fname)
    return fdcc_frp_timestamp,fdcc_frp,fdcc_frp_timestamp_aggregrate,fdcc_frp_aggregrate,fdcc_frp_err_aggregrate


def goes_plot(in_yyyymmdd,in_hhmmss,in_loc,in_lat,in_lon,in_goes):
    fig = plt.figure(figsize=(8,8))
    map_buffer = 6
    map_extent = [in_lon-map_buffer,in_lon+map_buffer,in_lat-map_buffer,in_lat+map_buffer]
    us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces',scale='10m',facecolor='none')
    axs = []
    axs.append(plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1, projection=ccrs.PlateCarree()))
    axs[0].set_extent(map_extent)
    gl = axs[0].gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1,color='gray',alpha=0.5,linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    axs[0].add_feature(us_states,edgecolor='k',zorder=100)
    mcmipc,geocolor,img_proj,img_extent,fname = get_goes_datetime('ABI-L2-MCMIPC', in_yyyymmdd, in_hhmmss, in_goes)        
    if mcmipc!='fuck':
        # --- plot GOES
        axs[0].imshow(np.flipud(geocolor),transform=img_proj,extent=img_extent,origin='lower',interpolation='nearest')
        axs[0].set_title(in_goes.upper()+' '+get_date_from_goes_file(fname)[0][:-3],fontsize=18,loc='left',y=0.99)
    # --- get GOES: cloud top temperature
    achtm,_,img_proj,img_extent,fname = get_goes_datetime('ABI-L2-ACHTF', in_yyyymmdd, in_hhmmss, in_goes)  
    if achtm!='fuck':
        im = axs[0].contourf(achtm['lon'], achtm['lat'], np.where(achtm['DQF']==0, achtm['TEMP'], np.nan),
                             np.arange(200,245,5),cmap='jet_r',alpha=0.6,transform=ccrs.PlateCarree())
    def add_colorbar(in_ax,in_im,in_display,in_ticks):
        axins = inset_axes(in_ax,width="25%",height="1%",bbox_to_anchor=(-0.05,-0.9,1,1),bbox_transform=in_ax.transAxes)
        cbar = plt.colorbar(in_im, cax=axins, orientation="horizontal",ticks=in_ticks)
        cbar.set_label(in_display, fontsize=12, color='w', labelpad=4)
        cbar.ax.tick_params(labelsize=12, color='w', labelcolor='w')
        cbar.outline.set_edgecolor('w')
        cbar.ax.invert_xaxis()
    add_colorbar(axs[0],im,'cloud top temperature (K)',[240,220,200])
    # --- add site info
    axs[0].plot(in_lon,in_lat,'C1*',ms=15,mec='w',mew=0.5,transform=ccrs.PlateCarree(),zorder=100)
    axs[0].text(in_lon,in_lat,'  '+in_loc,color='w',fontsize=15,transform=ccrs.PlateCarree(),zorder=100)
    del mcmipc,geocolor,img_proj,img_extent,achtm
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95, hspace=0.1, wspace=0.1)
    fname = OutputDir+'anim_%s_%s_%s_%s.png' % (in_goes,in_loc.replace(' ',''),in_yyyymmdd,in_hhmmss)
    plt.savefig(fname, dpi=150)
    plt.close()
    gc.collect()
    return fname


# In[10]:


# # ==============================================
# # GOES tru color image + cloud top: previous day
# # ==============================================
# def make_goes_animation(in_loc,in_lat,in_lon,in_frame_start_yyyymmdd,in_frame_start_hhmmss,in_goes):
#     y,m,d = int(in_frame_start_yyyymmdd[0:4]),int(in_frame_start_yyyymmdd[4:6]),int(in_frame_start_yyyymmdd[6:8])
#     hh,mm,ss = int(in_frame_start_hhmmss[:2]),int(in_frame_start_hhmmss[2:4]),int(in_frame_start_hhmmss[4:6])
#     frame_start_datetime = datetime(y,m,d,hh,mm,ss)
#     anim_fname_list = []
#     for i in range(17):
#         gc.collect()
#         frame_dt = frame_start_datetime + i*timedelta(seconds=1800)
#         in_yyyymmdd = str(frame_dt).split(' ')[0].replace('-','')
#         in_hhmmss = str(frame_dt).split(' ')[1].replace(':','')
#         fname = goes_plot(in_yyyymmdd,in_hhmmss,in_loc,in_lat,in_lon,in_goes)
#         anim_fname_list.append(fname)
#         print(fname)
#     # --- now make animation. imageio may fuck up color. use convert instead
#     fname = OutputDir+'anim_%s_%s_init_%s.gif' % (in_goes,in_loc.replace(' ',''),in_frame_start_yyyymmdd)
#     os.system('convert -delay 80 %sanim_%s_%s_????????_??????.png %s' % (OutputDir,in_goes,in_loc.replace(' ',''),fname))
#     os.system('rm %sanim_%s_%s_????????_??????.png' % (OutputDir,in_goes,in_loc.replace(' ','')))
#     return fname


# # # --- this takes about 15 min
# # goes_prevday_anim_fname = make_goes_animation(prod_loc, prod_lat, prod_lon, 
# #                                               prod_goes_yyyymmdd, prod_goes_start_hhmmss, which_goes)

# # print(goes_prevday_anim_fname)


# In[ ]:





# In[11]:


# ==========
# HRRR funks
# ==========
def get_hrrr(in_yyyymmdd, in_init_hour, in_forecast_hour):    
    f = 'noaa-hrrr-bdp-pds/hrrr.%s/conus/hrrr.t%02dz.wrfnatf%02d.grib2' % (in_yyyymmdd, in_init_hour, in_forecast_hour)
    fname = f.split('/')[1] +'.'+ f.split('/')[-1]
    local_dir = DataDump+'hrrr_fcst_temp/'
    if (os.path.exists(local_dir+fname)==True):
        ds = xr.open_dataset(local_dir+fname,engine='pynio')
        return f, ds
    else:
        # fs = s3fs.S3FileSystem(anon=True)  # don't fucking do this for each fucking file inquiry!
        if fs.exists(f):
            if (os.path.exists(local_dir+fname)==False): 
                print('*** now downloading HRRR: %s' % (fname))
                fs.get(f, local_dir+fname)
                ds = xr.open_dataset(local_dir+fname,engine='pynio')
                return f, ds
        else:
            print('*** file does not exist:', f)
            return 'fuck','fuck'
        

def hrrr_datetime_formatted(in_hrrr_fname):
    tmp = in_hrrr_fname.split('/')[1].replace('hrrr.','')
    init_h = int(in_hrrr_fname.split('/')[3].split('.')[1][1:3])
    fcst_h = int(in_hrrr_fname.split('/')[3].split('.')[2][-2:])
    init_datetime = datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8])) + timedelta(hours=init_h)
    valid_datetime = init_datetime + timedelta(hours=fcst_h)
    return 'HRRR PBLH, 850mb wind\nvalid ' + str(valid_datetime)[:-6] + 'Z (forecast hour: %d)' % (fcst_h)

def hrrr_datetime(in_hrrr_fname):
    if (in_hrrr_fname!='fuck'):
        tmp = in_hrrr_fname.split('/')[1].replace('hrrr.','')
        init_h = int(in_hrrr_fname.split('/')[3].split('.')[1][1:3])
        fcst_h = int(in_hrrr_fname.split('/')[3].split('.')[2][-2:])
        init_datetime = datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8])) + timedelta(hours=init_h)
        valid_datetime = init_datetime + timedelta(hours=fcst_h)
        return valid_datetime
    else: return ''


def make_hrrr_anim(in_loc, in_lat, in_lon,
                   in_hrrr_init_yyyymmdd,in_hrrr_init_hr,in_hrrr_fcst_hr_list):
    # --- download/load HRRR files
    list_hrrr_f, list_hrrr_ds = [],[]
    for hrrr_fcst_hr in in_hrrr_fcst_hr_list:
        y,m,d = int(in_hrrr_init_yyyymmdd[:4]),int(in_hrrr_init_yyyymmdd[4:6]),int(in_hrrr_init_yyyymmdd[6:8])
        hrrr_init_datetime = datetime(y,m,d,in_hrrr_init_hr)
        hrrr_dt = hrrr_init_datetime + timedelta(seconds=hrrr_fcst_hr*3600.)
        in_yyyymmdd = str(hrrr_dt).split(' ')[0].replace('-','')
        print(hrrr_dt)
        # hrrr_f, hrrr_ds = get_hrrr(in_yyyymmdd, in_hrrr_init_hr, hrrr_fcst_hr)
        hrrr_f, hrrr_ds = get_hrrr(in_hrrr_init_yyyymmdd, in_hrrr_init_hr, hrrr_fcst_hr)
        list_hrrr_f.append(hrrr_f)
        list_hrrr_ds.append(hrrr_ds)
    dt_list = []
    list_pblh, list_prf_pres, list_prf_temp, list_prf_gph, list_prf_spfh, list_prf_u, list_prf_v = [],[],[],[],[],[],[]
    for i,(hrrr_f, hrrr_ds) in enumerate(zip(list_hrrr_f, list_hrrr_ds)):
        gc.collect()    
        print('--- processing profiles: %d/%d' % (i+1,len(list_hrrr_f)))
        dt_list.append(hrrr_datetime(hrrr_f))
        pblh = -999
        map_buffer_small = 1
        if hrrr_f!='fuck':
            keys = list(hrrr_ds.keys())
            mask = (hrrr_ds['gridlon_0'].data>=in_lon-map_buffer_small-0.5) & (hrrr_ds['gridlon_0'].data<=in_lon+map_buffer_small+0.5) \
                 & (hrrr_ds['gridlat_0'].data>=in_lat-map_buffer_small-0.5) & (hrrr_ds['gridlat_0'].data<=in_lat+map_buffer_small+0.5)
            # --- get mean PBLH and mean profiles
            nlev = hrrr_ds['PRES_P0_L105_GLC0'].shape[0]-23
            prf_pres, prf_temp, prf_gph, prf_spfh, prf_u, prf_v = [],[],[],[],[],[]
            if 'HPBL_P0_L1_GLC0' in keys: pblh = np.nanmean(np.where(mask, hrrr_ds['HPBL_P0_L1_GLC0'], np.nan))/1000.
            if 'PRES_P0_L105_GLC0' in keys: prf_pres = [np.nanmean(np.where(mask,hrrr_ds['PRES_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            if 'TMP_P0_L105_GLC0' in keys: prf_temp = [np.nanmean(np.where(mask,hrrr_ds['TMP_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            if 'HGT_P0_L105_GLC0' in keys: prf_gph = [np.nanmean(np.where(mask,hrrr_ds['HGT_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            if 'SPFH_P0_L105_GLC0' in keys: prf_spfh = [np.nanmean(np.where(mask,hrrr_ds['SPFH_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            if 'UGRD_P0_L105_GLC0' in keys: prf_u = [np.nanmean(np.where(mask,hrrr_ds['UGRD_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            if 'VGRD_P0_L105_GLC0' in keys: prf_v = [np.nanmean(np.where(mask,hrrr_ds['VGRD_P0_L105_GLC0'][l,:,:],np.nan)) for l in range(nlev)]
            prf_height_km = np.array(prf_gph)*9.80665*6371.*1000./(9.80665*6371.*1000.-np.array(prf_gph)*9.80665)/1000.
            list_pblh.append(pblh)
            list_prf_pres.append(prf_pres)
            list_prf_temp.append(prf_temp)
            list_prf_gph.append(prf_gph)
            list_prf_spfh.append(prf_spfh)
            list_prf_u.append(prf_u)
            list_prf_v.append(prf_v)

    def plot_hrrr_pbl(tind):
        # --- setup plot
        fig = plt.figure(figsize=(6,8))
        map_buffer = 5
        map_extent = [in_lon-map_buffer,in_lon+map_buffer,in_lat-map_buffer,in_lat+map_buffer]
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_shp',scale='110m',facecolor='none')
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='10m',facecolor='none')
        us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces',scale='10m',facecolor='none')
        axs = [plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=3, projection=ccrs.PlateCarree()),
               plt.subplot2grid((4,1), (3,0), colspan=1, rowspan=1)]
        axs[0].set_extent(map_extent)
        axs[0].add_feature(us_states,edgecolor='k',zorder=100)
        gl = axs[0].gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=0.5,color='gray',alpha=0.5,linestyle='-')
        gl.top_labels,gl.right_labels = False,False
        axs[0].plot(in_lon,in_lat,'C1*',ms=20,mec='w',mew=0.5,transform=ccrs.PlateCarree(),zorder=100)
        axs[0].text(in_lon,in_lat,'  '+in_loc,color='k',fontsize=15,fontweight='bold',transform=ccrs.PlateCarree(),zorder=100)
        map_buffer_small = 1
        axins0 = []
        frac = float(1/len(list_hrrr_f))
        for i,(hrrr_f,pblh,prf_spfh,prf_u,prf_v) in enumerate(zip(list_hrrr_f,list_pblh,list_prf_spfh,list_prf_u,list_prf_v)):
            gc.collect()
            
            axins0.append(axs[-1].inset_axes([i*frac, 0, frac, 1.]))
            
#             axins0.append(inset_axes(axs[-1], width="7%", height="100%", loc="lower left",
#                                      bbox_to_anchor=(0.047+0.119*i,-0.04,1,1),bbox_transform=axs[-1].transAxes))
            # --- now plot the fucking profiles
            dt = hrrr_datetime(hrrr_f)
            if hrrr_f!='fuck':
                # --- add wind barbs
                wind_every_N = 2
                axs[-1].barbs([dt for v in prf_height_km][::wind_every_N],prf_height_km[::wind_every_N],
                              np.array(prf_u[::wind_every_N])*1.94384,np.array(prf_v[::wind_every_N])*1.94384,length=5,color='k')
                # --- add small moisture profiles
                axins0[-1].plot(prf_spfh,prf_height_km,'C1-',label='specific humidity')
                axins0[-1].hlines(np.nanmedian(pblh),xmin=axins0[-1].get_xlim()[0],xmax=axins0[-1].get_xlim()[1],label='PBLH')
            # --- configure the profile plot
            axs[-1].set_xlim([dt_list[0]-timedelta(hours=0.5),dt_list[-1]+timedelta(hours=0.9)])
            axs[-1].set_xticks(dt_list)
            axs[-1].set_ylim([0,8])
            axs[-1].set_ylabel('altitude (km, AGL)')
            axs[-1].xaxis.set_major_formatter(DateFormatter("%b-%d\n%HZ"))
            axs[-1].set_title('HRRR soundings',y=0.99,x=0.,ha='left',fontsize=15,fontweight='bold')
            # --- configure inserts
            for axin in axins0:
                axin.set_facecolor('none')  #  axin.set_facecolor('C1') # 
                axin.grid('none')
                axin.set_yticks([])
                axin.set_xticks([])
                for s in ['left','bottom','right','top']: axin.spines[s].set_color('none')
                axin.set_ylim([0,8])
        def add_colorbar(in_im, in_ax, in_bbox, in_label, in_ticks, in_height='100%'):
            axins0 = inset_axes(in_ax,width='4%',height=in_height,bbox_to_anchor=in_bbox,bbox_transform=in_ax.transAxes)
            axins0.grid(False)
            cbar0 = fig.colorbar(in_im, cax=axins0, orientation='vertical', extend='both', ticks=in_ticks)
            cbar0.set_label(in_label,fontsize=12,color='k')
            cbar0.ax.tick_params(axis='y',labelsize=12,colors='k')
            cbar0.ax.set_yticklabels(in_ticks)
        hrrr_f, hrrr_ds = list_hrrr_f[tind], list_hrrr_ds[tind]
        gc.collect() 
        if hrrr_f!='fuck':
            mask = (hrrr_ds['gridlon_0'].data>=map_extent[0]-1) & (hrrr_ds['gridlon_0'].data<=map_extent[1]+1) \
                 & (hrrr_ds['gridlat_0'].data>=map_extent[2]-1) & (hrrr_ds['gridlat_0'].data<=map_extent[3]+1)            
            # --- plot PBLH
            if 'HPBL_P0_L1_GLC0' in list(hrrr_ds.keys()):
                im2 = axs[0].contourf(hrrr_ds['gridlon_0'],hrrr_ds['gridlat_0'],
                                      np.where(mask, hrrr_ds['HPBL_P0_L1_GLC0'], np.nan)/1000.,
                                      np.arange(0,4,0.5),cmap='Spectral_r',extend='max',transform=ccrs.PlateCarree())
                axs[0].set_title(hrrr_datetime_formatted(hrrr_f),y=0.99,x=0.,ha='left',fontsize=15,fontweight='bold')
                add_colorbar(im2, axs[0], (0.1,0.02,1,1), 'PBLH (km)', [0,1,2,3])
            # --- plot winds
            if ('PRES_P0_L105_GLC0' in list(hrrr_ds.keys())) & ('UGRD_P0_L105_GLC0' in list(hrrr_ds.keys())) & ('VGRD_P0_L105_GLC0' in list(hrrr_ds.keys())):
                mean_press_hPa = np.mean(np.mean(hrrr_ds['PRES_P0_L105_GLC0'].data, axis=1), axis=1) /100.
                ref_press_ind = abs(mean_press_hPa-850.).argmin()
                skip_every_N = 10
                rap_u = np.where(mask, hrrr_ds['UGRD_P0_L105_GLC0'][ref_press_ind,:,:], np.nan)[::skip_every_N,::skip_every_N]
                rap_v = np.where(mask, hrrr_ds['VGRD_P0_L105_GLC0'][ref_press_ind,:,:], np.nan)[::skip_every_N,::skip_every_N]
                axs[0].streamplot(hrrr_ds['gridlon_0'][::skip_every_N,::skip_every_N].data.ravel(), 
                                  hrrr_ds['gridlat_0'][::skip_every_N,::skip_every_N].data.ravel(), 
                                  rap_u.ravel(),rap_v.ravel(),density=0.6,
                                  linewidth=1,color='k',transform=ccrs.PlateCarree())
            axins0[tind].set_facecolor('C3')
            axins0[tind].patch.set_alpha(0.4)
        del hrrr_ds
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.86, hspace=0.05, wspace=0.05)
        imgname = OutputDir+'anim_HRRR_PBL_%s_%s_%03d.jpeg' % (in_loc.replace(' ',''),in_yyyymmdd,tind)
        plt.savefig(imgname, dpi=150)
        plt.close()
        return imgname
    # --- loop over all frames then assemble gif
    for i in range(len(list_hrrr_f)): 
        plot_hrrr_pbl(i)
        ### break
    fname = OutputDir+'anim_HRRR_PBL_%s_%s.gif' % (in_loc.replace(' ',''),in_yyyymmdd)
    os.system('convert -delay 80 %sanim_HRRR_PBL_%s_%s_*.jpeg %s' % (OutputDir,in_loc.replace(' ',''),in_yyyymmdd,fname))
    os.system('rm %sanim_HRRR_PBL_%s_%s_*.jpeg' % (OutputDir,in_loc.replace(' ',''),in_yyyymmdd))
    return fname


# hrrr_fcst_anim_fname = make_hrrr_anim(prod_loc, prod_lat, prod_lon,
#                                       hrrr_init_yyyymmdd,hrrr_init_hr,hrrr_fcst_hr_list)


# In[ ]:





# In[12]:


# ==========
# HRRR funks
# ==========
def hrrr_datetime_formatted2(in_prod,in_hrrr_fname):
    tmp = in_hrrr_fname.split('/')[1].replace('hrrr.','')
    init_h = int(in_hrrr_fname.split('/')[3].split('.')[1][1:3])
    fcst_h = int(in_hrrr_fname.split('/')[3].split('.')[2][-2:])
    init_datetime = datetime(int(tmp[:4]),int(tmp[4:6]),int(tmp[6:8])) + timedelta(hours=init_h)
    valid_datetime = init_datetime + timedelta(hours=fcst_h)
    return in_prod+': valid ' + str(valid_datetime)[:-6] + 'Z (forecast hour: %d)' % (fcst_h)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval,b=maxval),
                                                                   cmap(np.linspace(minval,maxval,n)))
    return new_cmap


def make_hrrr_anim_cloud(in_loc, in_lat, in_lon,
                         in_hrrr_init_yyyymmdd,in_hrrr_init_hr,in_hrrr_fcst_hr_list):
    # --- download/load HRRR files
    list_hrrr_f, list_hrrr_ds = [],[]
    dt_list,mean_cloud_covers = [],[]
    for hrrr_fcst_hr in in_hrrr_fcst_hr_list:
        y,m,d = int(in_hrrr_init_yyyymmdd[:4]),int(in_hrrr_init_yyyymmdd[4:6]),int(in_hrrr_init_yyyymmdd[6:8])
        hrrr_init_datetime = datetime(y,m,d,in_hrrr_init_hr)
        hrrr_dt = hrrr_init_datetime + timedelta(seconds=hrrr_fcst_hr*3600.)
        in_yyyymmdd = str(hrrr_dt).split(' ')[0].replace('-','')
        print(hrrr_dt)
        # hrrr_f, hrrr_ds = get_hrrr(in_yyyymmdd, in_hrrr_init_hr, hrrr_fcst_hr)
        hrrr_f, hrrr_ds = get_hrrr(in_hrrr_init_yyyymmdd, in_hrrr_init_hr, hrrr_fcst_hr)
        list_hrrr_f.append(hrrr_f)
        list_hrrr_ds.append(hrrr_ds)
        map_buffer_small = 1.
        if hrrr_f!='fuck':
            keys = list(hrrr_ds.keys())
            mask = (hrrr_ds['gridlon_0'].data>=in_lon-map_buffer_small-0.5) & (hrrr_ds['gridlon_0'].data<=in_lon+map_buffer_small+0.5) \
                 & (hrrr_ds['gridlat_0'].data>=in_lat-map_buffer_small-0.5) & (hrrr_ds['gridlat_0'].data<=in_lat+map_buffer_small+0.5)
            # --- get mean cloud covers
            avg_tmp = []
            for var in ['LCDC_P0_L214_GLC0','MCDC_P0_L224_GLC0','HCDC_P0_L234_GLC0']:
                if var in keys: avg_tmp.append(np.nanmean(np.where(mask, hrrr_ds[var], np.nan)))
                else: avg_tmp.append(np.nan)
            mean_cloud_covers.append(avg_tmp)
            dt_list.append(hrrr_datetime(hrrr_f))
        else: 
            mean_cloud_covers.append([np.nan,np.nan,np.nan])
            dt_list.append(np.nan)    
    mean_cloud_covers = np.array(mean_cloud_covers)
    
    def plot_hrrr_cloud(tind):
        # --- setup plot
        map_buffer = 5
        map_extent = [in_lon-map_buffer,in_lon+map_buffer,in_lat-map_buffer,in_lat+map_buffer]
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='110m',facecolor='none')
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_shp',scale='110m',facecolor='none')
        # us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lakes_shp',scale='10m',facecolor='none')
        us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces',scale='10m',facecolor='none')

        img = plt.imread('/home/swang/my_python/land_ocean_ice_8192.png')
        fig = plt.figure(figsize=(8,4))
        grid = (3,6)
        axs = []
        axs.append(plt.subplot2grid(grid, (0,0), colspan=2, rowspan=2, projection=ccrs.PlateCarree()))
        axs.append(plt.subplot2grid(grid, (0,2), colspan=2, rowspan=2, projection=ccrs.PlateCarree()))
        axs.append(plt.subplot2grid(grid, (0,4), colspan=2, rowspan=2, projection=ccrs.PlateCarree()))
        axs.append(plt.subplot2grid(grid, (2,0), colspan=6, rowspan=1))
        
        for i in [0,1,2]:
            axs[i].set_extent(map_extent)
            axs[i].imshow(img, origin='upper', extent=(-180,180,-90,90), alpha=0.9, zorder=-1, transform=ccrs.PlateCarree())
            axs[i].add_feature(us_states,edgecolor='k',zorder=100)
            axs[i].plot(in_lon,in_lat,'C1*',ms=20,mec='w',mew=0.5,transform=ccrs.PlateCarree(),zorder=100)
            axs[i].text(in_lon,in_lat,'  '+in_loc,color='k',fontsize=15,fontweight='bold',transform=ccrs.PlateCarree(),zorder=100)
            
        hrrr_f, hrrr_ds = list_hrrr_f[tind], list_hrrr_ds[tind]
        gc.collect() 
        
        def add_colorbar(in_im, in_ax, in_bbox, in_label, in_ticks):
            axins0 = inset_axes(in_ax,width="5%",height="100%",bbox_to_anchor=in_bbox,bbox_transform=in_ax.transAxes)
            cbar0 = fig.colorbar(in_im, cax=axins0, orientation="vertical", ticks=in_ticks) # extend='both', 
            cbar0.set_label(in_label,fontsize=12,color='k')
            cbar0.ax.tick_params(labelsize=12)
            cbar0.ax.tick_params(size=0)
            cbar0.ax.tick_params(axis='y', colors='k')
        
        cloudcolors = plt.cm.BuPu(np.linspace(0.2, 0.6, 3))
        if hrrr_f!='fuck':
            # --- add maps
            mask = (hrrr_ds['gridlon_0'].data>=map_extent[0]-1) & (hrrr_ds['gridlon_0'].data<=map_extent[1]+1) \
                 & (hrrr_ds['gridlat_0'].data>=map_extent[2]-1) & (hrrr_ds['gridlat_0'].data<=map_extent[3]+1)            
            fig.suptitle(hrrr_datetime_formatted2('HRRR cloud cover',hrrr_f),x=0.05,y=0.99,ha='left',fontsize=15,fontweight='bold') # y=0.99,
            # --- plot cloud cover
            for n,(var,varname) in enumerate(zip(['LCDC_P0_L214_GLC0','MCDC_P0_L224_GLC0','HCDC_P0_L234_GLC0'],
                                                 ['low cloud','mid cloud','high cloud'])):
                if var in list(hrrr_ds.keys()):
                    im = axs[n].contourf(hrrr_ds['gridlon_0'],hrrr_ds['gridlat_0'],
                                          np.where(mask, hrrr_ds[var], np.nan),
                                          np.arange(5,110,10),cmap=truncate_colormap(plt.get_cmap('Greys_r'), 0.45, 0.8),
                                         extend='neither',transform=ccrs.PlateCarree())
                    axs[n].set_title(varname,y=0.01,x=0.05,ha='left',va='bottom',fontsize=13,fontweight='bold',
                                     bbox=dict(facecolor='w', alpha=0.4))
                    
                    
            add_colorbar(im,axs[2],(0.1,0.03,1,1),'cloud cover (%)', [10,50,100])
            # --- add average
            axs[-1].bar(dt_list,mean_cloud_covers[:,0],width=0.05,facecolor=cloudcolors[0],edgecolor='w',label='low',zorder=3)
            axs[-1].bar(dt_list,mean_cloud_covers[:,1],bottom=mean_cloud_covers[:,0],width=0.05,facecolor=cloudcolors[1],edgecolor='w',label='mid',zorder=2)
            axs[-1].bar(dt_list,mean_cloud_covers[:,2],bottom=mean_cloud_covers[:,0]+mean_cloud_covers[:,1],edgecolor='w',width=0.05,label='high',zorder=1)
            axs[-1].xaxis.set_major_formatter(DateFormatter("%b-%d\n%HZ"))
            axs[-1].set_xticks(dt_list)
            axs[-1].set_ylabel('cloud cover (%)')
            axs[-1].legend(bbox_to_anchor=(1, 1.05),loc='upper left')
            # --- add box indicating current hour
            axs[-1].bar(dt_list[tind],mean_cloud_covers[tind,0]+mean_cloud_covers[tind,1]+mean_cloud_covers[tind,2],facecolor='None',edgecolor='red',linewidth=1,width=0.05,zorder=5)
        del hrrr_ds
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.88, hspace=0.05, wspace=0.05)
        imgname = OutputDir+'anim_HRRR_clouds_%s_%s_%03d.jpeg' % (in_loc.replace(' ',''),in_yyyymmdd,tind)
        plt.savefig(imgname, dpi=150)
        plt.close()
        return imgname
    # --- loop over all frames then assemble gif
    for i in range(len(list_hrrr_f)): 
        plot_hrrr_cloud(i)
        ### break
    fname = OutputDir+'anim_HRRR_clouds_%s_%s.gif' % (in_loc.replace(' ',''),in_yyyymmdd)
    os.system('convert -delay 80 %sanim_HRRR_clouds_%s_%s_*.jpeg %s' % (OutputDir,in_loc.replace(' ',''),in_yyyymmdd,fname))
    os.system('rm %sanim_HRRR_clouds_%s_%s_*.jpeg' % (OutputDir,in_loc.replace(' ',''),in_yyyymmdd))
    return fname


# hrrr_fcst_cloud_anim_fname = make_hrrr_anim_cloud(prod_loc, prod_lat, prod_lon,
#                                                   hrrr_init_yyyymmdd,hrrr_init_hr,hrrr_fcst_hr_list)

# hrrr_fcst_cloud_anim_fname


# In[ ]:





# In[13]:


# ========================================
# Quick HRRR eval using GOES: previous day
# ========================================
# --- Latest GOES
# Ch       WL
# 08       6.2: upper water vapor, useful for jet stream, hurrican track, storm forecasting, severe weather, etc
# 09       6.9
# 13      10.3: clear longwave IR: not strongly affected by water vapor, useful for clouds day and night

# --- GOES 11/12
# Ch      WL range             WL center
# 03       5.7690 to  7.3351     6.4835
# 04      10.2344 to 11.2397    10.7134

# --- HRRR variables
# SBT123_P0_L8_GLC0: Simulated brightness temperature for GOES 12, channel 3
# SBT124_P0_L8_GLC0: Simulated brightness temperature for GOES 12, channel 4
# SBT113_P0_L8_GLC0: Simulated brightness temperature for GOES 11, channel 3
# SBT114_P0_L8_GLC0: Simulated brightness temperature for GOES 11, channel 4

# # --- import colormap: https://cimss.ssec.wisc.edu/hrrrval/
# img = imread('cimss_cmap.png')
# cimss_cmap = LinearSegmentedColormap.from_list('cimss_cmap', img[0, :, :], N=280)

# --- import NESDIS colormap
# urllib.request.urlretrieve('https://www.star.nesdis.noaa.gov/goes/images/colorbars/ColorBar450Bands8-10_horz.png', 'ColorBar450Bands8-10_horz.png')
img = imread('/home/swang/my_python/ColorBar450Bands8-10_horz.png')
nesdis_goes_ch8_cmap = LinearSegmentedColormap.from_list('nesdis_goes_ch8_cmap', img[2, :, :], N=280)

# urllib.request.urlretrieve('https://www.star.nesdis.noaa.gov/goes/images/colorbars/ColorBar450Bands11-15_horz.png', 'ColorBar450Bands11-15_horz.png')
img = imread('/home/swang/my_python/ColorBar450Bands11-15_horz.png')
nesdis_goes_ch13_cmap = LinearSegmentedColormap.from_list('nesdis_goes_ch13_cmap', img[2, :, :], N=280)


def plot_goes_hrrr_eval(in_loc, in_lat, in_lon,
                        in_yyyymmdd, in_hhmmss, in_goes,
                        in_hrrr_init_hr, in_hrrr_fcst_hr):
    if in_goes=='goes16': 
        hrrr_sb = 'SBT123_P0_L8_GLC0'   # GOES-12 is east
        hrrr_lb = 'SBT124_P0_L8_GLC0'   # GOES-12 is east
    if (in_goes=='goes17') | (in_goes=='goes18'): 
        hrrr_sb = 'SBT113_P0_L8_GLC0'   # GOES-11 is west
        hrrr_lb = 'SBT114_P0_L8_GLC0'   # GOES-11 is west
    # --- download the data
    hrrr_f, hrrr_ds = get_hrrr(in_yyyymmdd, in_hrrr_init_hr, in_hrrr_fcst_hr)
    mcmipc,geocolor,img_proj,img_extent,fname = get_goes_datetime('ABI-L2-MCMIPC', in_yyyymmdd, in_hhmmss, in_goes)
    # --- now plot the shit
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    us_states = cfeature.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces',scale='50m',facecolor='none')    
    def add_colorbar(in_ax,in_im,in_display,in_ticks):
        axins = inset_axes(in_ax,width="3%",height="100%",bbox_to_anchor=(0.1,0.03,1,1),bbox_transform=in_ax.transAxes)
        cbar = plt.colorbar(in_im, cax=axins, orientation="vertical",ticks=in_ticks)
        cbar.set_label(in_display, fontsize=12, color='k', labelpad=4)
        cbar.ax.tick_params(labelsize=12, color='k', labelcolor='k')
        cbar.outline.set_edgecolor('k')
    # clev,cmap = np.arange(180,310,5),cimss_cmap
    clev_sb,cmap_sb = np.arange(273-93,273+7,1),nesdis_goes_ch8_cmap
    clev_lb,cmap_lb = np.arange(273-110,273+57,1),nesdis_goes_ch13_cmap
    if mcmipc!='fuck':
        # im0 = axs[0].contourf(mcmipc['lon'],mcmipc['lat'],mcmipc['CMI_C08'],clev,cmap=cmap,extend='neither',transform=ccrs.PlateCarree())
        mask = (mcmipc['lon'].data>=in_lon-15) & (mcmipc['lon'].data<=in_lon+15) & (mcmipc['lat'].data>=in_lat-15) & (mcmipc['lat'].data<=in_lat+15)
        z,x,y = np.where(mask,mcmipc['CMI_C08'].data,np.nan,),np.where(mask,mcmipc['lon'].data,np.nan,),np.where(mask,mcmipc['lat'].data,np.nan,)
        axs[0,0].contourf(x,y,z,clev_sb,cmap=cmap_sb,extend='neither',transform=ccrs.PlateCarree())
        axs[0,0].set_title(in_goes.upper()+r' band-08 (6.2$\mu$m)'+'\n'+get_date_from_goes_file(fname)[0][:-3],fontsize=12,loc='left')
        z,x,y = np.where(mask,mcmipc['CMI_C13'].data,np.nan,),np.where(mask,mcmipc['lon'].data,np.nan,),np.where(mask,mcmipc['lat'].data,np.nan,)
        axs[1,0].contourf(x,y,z,clev_lb,cmap=cmap_lb,extend='neither',transform=ccrs.PlateCarree())
        axs[1,0].set_title(in_goes.upper()+r' band-13 (10.3$\mu$m)'+'\n'+get_date_from_goes_file(fname)[0][:-3],fontsize=12,loc='left')
    if hrrr_ds!='fuck':
        # im1 = axs[1].contourf(hrrr_ds['gridlon_0'],hrrr_ds['gridlat_0'],hrrr_ds[hrrr_sb],clev,cmap=cmap,extend='both')
        # im1 = axs[1].pcolormesh(hrrr_ds['gridlon_0'],hrrr_ds['gridlat_0'],hrrr_ds[hrrr_sb],vmin=clev[0],vmax=clev[-1],cmap=cmap,shading='nearest',transform=ccrs.PlateCarree())
        mask = (hrrr_ds['gridlon_0'].data>=in_lon-15) & (hrrr_ds['gridlon_0'].data<=in_lon+15) & (hrrr_ds['gridlat_0'].data>=in_lat-15) & (hrrr_ds['gridlat_0'].data<=in_lat+15)
        z,x,y = np.where(mask,hrrr_ds[hrrr_sb].data,np.nan,),np.where(mask,hrrr_ds['gridlon_0'].data,np.nan,), np.where(mask,hrrr_ds['gridlat_0'].data,np.nan,)
        im01 = axs[0,1].contourf(x,y,z,clev_sb,cmap=cmap_sb,extend='neither')
        axs[0,1].set_title(r'HRRR simulated IR 6.5$\mu$m'+'\n'+hrrr_datetime_formatted(hrrr_f).split('\n')[1],fontsize=12,loc='left')
        add_colorbar(axs[0,1],im01,'Brightness temperature (K)',np.arange(180,290,10))
        z,x,y = np.where(mask,hrrr_ds[hrrr_lb].data,np.nan,),np.where(mask,hrrr_ds['gridlon_0'].data,np.nan,), np.where(mask,hrrr_ds['gridlat_0'].data,np.nan,)
        im11 = axs[1,1].contourf(x,y,z,clev_lb,cmap=cmap_lb,extend='neither')
        axs[1,1].set_title(r'HRRR simulated IR 10.7$\mu$m'+'\n'+hrrr_datetime_formatted(hrrr_f).split('\n')[1],fontsize=12,loc='left')
        add_colorbar(axs[1,1],im11,'Brightness temperature (K)',np.arange(170,330,10))
    for ax in axs.flatten():
        ax.set_extent([in_lon-10, in_lon+10, in_lat-10, in_lat+10])
        # ax.set_extent([-130,-50,20,50])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(us_states,edgecolor='k',zorder=100)
        ax.plot(in_lon,in_lat,'C1*',ms=15,mec='k',mew=0.5,transform=ccrs.PlateCarree(),zorder=100)
        ax.text(in_lon,in_lat,'  '+in_loc,color='k',fontsize=15,transform=ccrs.PlateCarree(),zorder=100)
    # --- misc
    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.87, hspace=0.1, wspace=0.05)
    fname = OutputDir+'goes_hrrr_eval_%s_%s_%s_%s.jpeg' % (in_goes,in_loc.replace(' ',''),in_yyyymmdd,in_hhmmss)
    plt.savefig(fname, dpi=200)
    plt.close()
    gc.collect()
    return fname


# plot_goes_hrrr_eval(prod_loc, prod_lat, prod_lon,'20230510', '190000', which_goes, 18, 1)

goes_hrrr_eval_fnames = []
for h in np.arange(15,23):
    print(h)
    goes_hrrr_eval_fnames.append(plot_goes_hrrr_eval(prod_loc, prod_lat, prod_lon,
                                                     prod_goes_yyyymmdd, '%02d0000' % (h), which_goes, h-1, 1))

hrrr_goes_eval_anim_fname = OutputDir+'anim_goes_hrrr_eval_%s_%s_%s.gif' % (which_goes,prod_loc.replace(' ',''),prod_goes_yyyymmdd)
os.system('convert -delay 80 %sgoes_hrrr_eval_%s_%s_%s_*.jpeg %s' % (OutputDir,which_goes,prod_loc,prod_goes_yyyymmdd,hrrr_goes_eval_anim_fname))

for f in goes_hrrr_eval_fnames: os.system('rm %s' % (f))


# In[ ]:





# In[14]:


# # =================================
# # ACARS profiles for given airports
# # =================================
# def get_acars(ind=-1):
#     # --- this is the publicly available MADIS ACARS profiles. Get most recent archive only
#     url = 'https://madis-data.ncep.noaa.gov/madisPublic1/data/point/acarsProfiles/netcdf/'
#     response = requests.get(url)
#     if response.ok: response_text = response.text
#     soup = BeautifulSoup(response_text, 'html.parser')
#     # --- geta list of files, then load the latest
#     sounds_flist = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.gz')]
#     f = sounds_flist[ind]
#     fname = 'acars_' + f.split('/')[-1]
#     try:
#         # --- download the data
#         urllib.request.urlretrieve(f, DataDump + fname)
#         # --- unpack the data
#         os.system('gunzip -f ' + DataDump + fname)
#     #     # --- remove the gz file
#     #     if (os.path.exists(DataDump + fname)): os.system('rm ' + DataDump + fname)
#         # --- load netCDF
#         acars = xr.open_dataset(DataDump + fname.replace('.gz',''), engine='netcdf4')
#         # --- get a list of airports
#         airport_list = [str(apid.data).replace('b','').replace("'",'') for apid in acars['profileAirport']]
#         print('all airports: ', airport_list)
#         return acars, airport_list
#     except:
#         return 'fuck','fuck'


# def plot_acars_profiles(in_ds, in_sounding_id, in_airport_id):
#     # --- calculate pressure, theta, specific humidity
#     altitude_m_asl = in_ds['altitude'][in_sounding_id,:].data
#     pressure_Pa = 101325. * (1. - (0.0065 * altitude_m_asl / 288.15))**((9.80665 * 0.02896) / (8.314 * 0.0065))
#     surface_pressure_Pa = 101325.* (1.-(0.0065*in_ds['elevation'][in_sounding_id].data/288.15))**((9.80665*0.02896)/(8.314*0.0065))
#     theta_K = potential_temperature(pressure_Pa*units.Pa, in_ds['temperature'][in_sounding_id,:].data*units.K).to('K')
#     specific_humidity_g_kg = specific_humidity_from_dewpoint(pressure_Pa*units.Pa,in_ds['dewpoint'][in_sounding_id,:].data*units.K).to('g/kg')
#     # --- calculate u/v
#     u_m_s = -in_ds['windSpeed'][in_sounding_id,:].data * np.sin(np.radians(in_ds['windDir'][in_sounding_id,:].data))
#     v_m_s = -in_ds['windSpeed'][in_sounding_id,:].data * np.cos(np.radians(in_ds['windDir'][in_sounding_id,:].data))
#     # --- set up plot
#     fig = plt.figure(figsize=(6,4))
#     axs = []
#     axs.append(plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2))
#     axs.append(plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1, projection=ccrs.PlateCarree()))
#     axs.append(plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1, projection=ccrs.PlateCarree()))
#     # --- plot specific humidity
#     axs[0].plot(specific_humidity_g_kg,altitude_m_asl/1000,'C0')
#     axs[0].set_xlabel('specific humidity (g kg$^{-1}$)',fontsize=11)
#     axs[0].set_ylim([0,8])
#     axs[0].set_ylabel('altitude (km, ASL)',fontsize=11)
#     xmax = 1.2*np.nanmax(specific_humidity_g_kg)
#     axs[0].hlines(in_ds['elevation'][in_sounding_id].data/1000.,xmin=0,xmax=xmax,color='k')
#     axs[0].set_xlim([0,xmax])
#     # --- plot wind barbs
#     axs[0].barbs([0.9*xmax for v in altitude_m_asl], altitude_m_asl/1000., u_m_s*1.94384, v_m_s*1.94384)
#     # --- plot theta
#     axs0top = axs[0].twiny()
#     axs0top.plot(theta_K,altitude_m_asl/1000,'C1')
#     axs0top.set_xlabel(r'potential temperature (K)',fontsize=11)
#     for i in [1,2]:
#         axs[i].plot(in_ds['longitude'][in_sounding_id],in_ds['latitude'][in_sounding_id],'C1*',ms=6,transform=ccrs.PlateCarree())
#         axs[i].text(in_ds['longitude'][in_sounding_id],in_ds['latitude'][in_sounding_id],'  '+in_airport_id,fontsize=10,
#                     color='C1',transform=ccrs.PlateCarree(),zorder=100)
#         axs[i].stock_img()
#         axs[i].set_ylim([20,60])
#         axs[i].set_xlim([-130,-50])
#     # --- misc
#     axs[2].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='wheat'))
#     axs[2].add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='lightskyblue'))
#     gl = axs[2].gridlines(draw_labels=True,dms=False,x_inline=False,y_inline=False,linewidth=0,color='gray',alpha=0,linestyle='-')
#     gl.top_labels = False
#     gl.right_labels = False
#     # --- plot trajectory
#     axs[2].plot(in_ds['trackLon'][in_sounding_id,:],in_ds['trackLat'][in_sounding_id,:],'k-',lw=2)
#     axs[2].set_ylim([np.nanmin(in_ds['trackLat'][in_sounding_id,:])-0.2,np.nanmax(in_ds['trackLat'][in_sounding_id,:])+0.2])
#     axs[2].set_xlim([np.nanmin(in_ds['trackLon'][in_sounding_id,:])-0.2,np.nanmax(in_ds['trackLon'][in_sounding_id,:])+0.2])
#     axs[0].grid(False)
#     axs0top.grid(False)
#     axs[0].grid(which='major', axis='y', linestyle='--')
#     for ax,c in zip([axs[0],axs0top],['C0','C1']):
#         ax.spines['top'].set_color(c)
#         ax.spines['bottom'].set_color(c)
#         ax.xaxis.label.set_color(c)
#         ax.tick_params(axis='x', colors=c)
#     title = 'MADIS ACARS profile: '
#     title = title + str(np.datetime_as_string(in_ds['profileTime'][in_sounding_id].data, unit='m')).replace('T',' ') + 'Z '
#     title = title + in_airport_id
#     if in_ds['profileType'][in_sounding_id].data==-1.: title = title + ' descending'
#     else: title = title + ' ascending'
#     fig.suptitle(title, fontsize=12)
#     plt.subplots_adjust(top=0.8, bottom=0.12, left=0.1, right=0.97, hspace=0.35, wspace=0.15)
#     acars_fname = OutputDir+title.replace(' ','_').replace(':','').replace('profile_','')+'.jpeg'
#     plt.savefig(acars_fname, dpi=200)
#     plt.close()
#     return acars_fname

# # # --- if no profiles for current hour, go back up to 6 hours (bit of a stretch)
# # list_airpot_id,list_acars_fname = [],[]
# # for airport in airports_of_interest:
# #     for acars_time_ind in [-1,-2,-3,-4,-5,-6]:
# #         acars, airport_list = get_acars(ind=acars_time_ind)
# #         if acars!='fuck':
# #             if airport in airport_list:
# #                 print(airport, acars_time_ind)
# #                 sounding_id = [i for i,apid in enumerate(airport_list) if apid.replace(' ','')==airport]
# #                 # print(airport_list)
# #                 print('%s: %d profile(s)' % (airport, len(sounding_id)))
# #                 for s in sounding_id:
# #                     print(airport, s)
# #                     acars_fname = plot_acars_profiles(acars, s, airport)
# #                     list_acars_fname.append(acars_fname)
# #                     list_airpot_id.append(airport)
# #                 break


# In[15]:


# def get_acars_datetime(in_yyyymmdd, in_hhmm):
#     # --- this is the publicly available MADIS ACARS profiles, recent archive only
#     url = 'https://madis-data.ncep.noaa.gov/madisPublic1/data/point/acarsProfiles/netcdf/'
#     response = requests.get(url)
#     if response.ok: response_text = response.text
#     soup = BeautifulSoup(response_text, 'html.parser')
#     # --- geta list of files, then load the latest
#     sounds_flist = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.gz')]
#     f = [f for f in sounds_flist if (in_yyyymmdd in f) & (in_hhmm in f)][0]
#     fname = 'acars_' + f.split('/')[-1]
#     # --- download the data
#     urllib.request.urlretrieve(f, DataDump + fname)
#     # --- unpack the data
#     os.system('gunzip -f ' + DataDump + fname)
# #     # --- remove the gz file
# #     if (os.path.exists(DataDump + fname)): os.system('rm ' + DataDump + fname)
#     # --- load netCDF
#     acars = xr.open_dataset(DataDump + fname.replace('.gz',''))
#     # --- get a list of airports
#     airport_list = [str(apid.data).replace('b','').replace("'",'') for apid in acars['profileAirport']]
#     # print('all airports: ', airport_list)
#     return acars, airport_list
            

# def plot_acars_profiles_eval_HRRR(in_ds, in_sounding_id, in_airport_id, in_hrrr_ds, in_airport, in_yyyymmdd, in_hhmm, in_hrrr_fcst_hr):
#     # --- calculate pressure, theta, specific humidity
#     altitude_m_asl = in_ds['altitude'][in_sounding_id,:].data
#     pressure_Pa = 101325. * (1. - (0.0065 * altitude_m_asl / 288.15))**((9.80665 * 0.02896) / (8.314 * 0.0065))
#     surface_pressure_Pa = 101325.* (1.-(0.0065*in_ds['elevation'][in_sounding_id].data/288.15))**((9.80665*0.02896)/(8.314*0.0065))
#     theta_K = potential_temperature(pressure_Pa*units.Pa, in_ds['temperature'][in_sounding_id,:].data*units.K).to('K')
#     specific_humidity_g_kg = specific_humidity_from_dewpoint(pressure_Pa*units.Pa,in_ds['dewpoint'][in_sounding_id,:].data*units.K).to('g/kg')
#     # --- calculate u/v
#     u_m_s = -in_ds['windSpeed'][in_sounding_id,:].data * np.sin(np.radians(in_ds['windDir'][in_sounding_id,:].data))
#     v_m_s = -in_ds['windSpeed'][in_sounding_id,:].data * np.cos(np.radians(in_ds['windDir'][in_sounding_id,:].data))
#     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(4.5,2))
#     axs[0].plot(theta_K,altitude_m_asl/1000,'C1o',mec='w',mew=0.5)
#     axs[1].plot(specific_humidity_g_kg,altitude_m_asl/1000,'C1o',mec='w',mew=0.5,label='ACARS')
#     axs[2].plot(u_m_s,altitude_m_asl/1000,'C1o',mec='w',mew=0.5)
#     axs[2].plot(v_m_s,altitude_m_asl/1000,'C0o',mec='w',mew=0.5)
#     # --- now process fucking HRRR
#     keys = list(hrrr_ds.keys())
#     hrrrinds = abs( in_hrrr_ds['gridlat_0'] - in_ds['latitude'].data[in_sounding_id] ) + abs( in_hrrr_ds['gridlon_0'] - in_ds['longitude'].data[in_sounding_id] )
#     hrrrinds_i,hrrrinds_j = np.unravel_index(hrrrinds.argmin(), hrrrinds.shape)
#     nlev = in_hrrr_ds['PRES_P0_L105_GLC0'].shape[0]-23
#     if 'PRES_P0_L105_GLC0' in keys: prf_pres = [np.median(hrrr_ds['PRES_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     if 'TMP_P0_L105_GLC0' in keys: prf_temp = [np.median(hrrr_ds['TMP_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     if 'HGT_P0_L105_GLC0' in keys: prf_gph = [np.median(hrrr_ds['HGT_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     if 'SPFH_P0_L105_GLC0' in keys: prf_spfh = [np.median(hrrr_ds['SPFH_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     if 'UGRD_P0_L105_GLC0' in keys: prf_u = [np.median(hrrr_ds['UGRD_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     if 'VGRD_P0_L105_GLC0' in keys: prf_v = [np.median(hrrr_ds['VGRD_P0_L105_GLC0'][l,hrrrinds_i-1:hrrrinds_i+1,hrrrinds_j-1:hrrrinds_j+1]) for l in range(nlev)]
#     prf_height_km = np.array(prf_gph)*9.80665*6371.*1000./(9.80665*6371.*1000.-np.array(prf_gph)*9.80665)/1000.
#     prf_pottemp = prf_temp*(100000./np.array(prf_pres))**(2./7.)
#     axs[0].plot(prf_pottemp,prf_height_km,'k-')
#     axs[1].plot(1000.*np.array(prf_spfh),prf_height_km,'k-',label='HRRR')
#     axs[2].plot(prf_u,prf_height_km,'C1-')
#     axs[2].plot(prf_v,prf_height_km,'C0-')
#     # --- misc
#     axs[1].legend()
#     axs[0].set_ylabel('altitiude (km)', fontsize=10)
#     axs[0].set_xlabel(r'$\theta$ (K)', fontsize=10)
#     axs[1].set_xlabel('Q$_v$ (g kg$^{-1}$)', fontsize=10)
#     axs[2].set_xlabel('u, v (m s$^{-1}$)', fontsize=10)
#     plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.99, hspace=0.2, wspace=0.2)
#     title = 'ACARS profile: %s, %s %sZ' % (in_airport, in_yyyymmdd, in_hhmm)
#     title = title + '\n' + 'HRRR forecast hour %d' % (in_hrrr_fcst_hr)
#     axs[0].set_title(title, fontsize=10, loc='left')
#     fname = OutputDir+'ACARS_HRRRR_%s_%s_%s.jpeg' % (in_airport, in_yyyymmdd, in_hhmm)
#     plt.savefig(fname, dpi=150)
#     plt.close()
#     return fname
    

# acars_hrrr_eval_fnames = []
# for h in range(14,24,1):  # that's about 10AM to 7PM EDT
#     print(h)
#     acars, airport_list = get_acars_datetime(prod_goes_yyyymmdd, '%02d00' % (h))
#     at_least_one_profile = False
#     for airport in airports_of_interest:
#         if (airport in airport_list) & (at_least_one_profile==False):
#             sounding_id = [i for i,apid in enumerate(airport_list) if apid.replace(' ','')==airport]
#             # print('%s, %02d hour' % (airport, h))
#             # print('souding ids: ', sounding_id)
#             # --- now get fucking HRRR
#             hrrr_time_init, hrrr_time_fcst = h-1, 1
#             hrrr_f, hrrr_ds = get_hrrr(prod_goes_yyyymmdd, hrrr_time_init, hrrr_time_fcst)
#             print(hrrr_f)
#             # --- now plot the profiles
#             if hrrr_ds!='fuck':
#                 for s in sounding_id:
#                     acars_hrrr_eval_fnames.append(plot_acars_profiles_eval_HRRR(acars, s, airport, hrrr_ds, airport, prod_goes_yyyymmdd, '%02d00' % (h), hrrr_time_fcst))
#             at_least_one_profile = True


# # --- upload to FTP
# for f in acars_hrrr_eval_fnames:
#     if (os.path.exists(f)): 
#         upload_to_FTP(f, ftp_dir_daily, f.split('/')[-1], ftp_server, ftp_username, ftp_password)


# In[9]:


# goes_prevday_anim_fname = '/home/swang/AEROMMA_tmp/anim_goes18_LA_init_20230521.gif'
# hrrr_goes_eval_anim_fname = '/home/swang/AEROMMA_tmp/anim_goes_hrrr_eval_goes18_LA_20230521.gif'
# acars_hrrr_eval_fnames = ['/home/swang/AEROMMA_tmp/ACARS_HRRRR_LAX_20230521_1400.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_LAX_20230521_1500.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_1600.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_LAX_20230521_1700.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_BUR_20230521_1800.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_BUR_20230521_1800.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_BUR_20230521_1800.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2000.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2000.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2100.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2200.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2300.jpeg', '/home/swang/AEROMMA_tmp/ACARS_HRRRR_SNA_20230521_2300.jpeg']

# goes_prevday_anim_fname = '/home/swang/AEROMMA_tmp/anim_goes16_NYC_init_20230523.gif'
# hrrr_goes_eval_anim_fname = '/home/swang/AEROMMA_tmp/anim_goes_hrrr_eval_goes16_NYC_20230523.gif'
# acars_hrrr_eval_fnames = ]


# In[17]:


# print('\n--- previous day: GOES img')
# print(goes_prevday_anim_fname)
print('\n--- previous day: HRRR vs GOES')
print(hrrr_goes_eval_anim_fname)
# print('\n--- previous day: HRRR vs ACARS')
# print(acars_hrrr_eval_fnames)


# In[ ]:





# In[ ]:





# In[21]:


# fname = OutputDir + f'previous_day_prod_{in_config_loc}_{prod_goes_yyyymmdd}'

# # --- write GOES previous day anim
# shit2write = f"goes_prevday_anim_fname = '{goes_prevday_anim_fname}'"

# # --- write GOES vs HRRR previous day anim
# shit2write = shit2write + '\n' + f"hrrr_goes_eval_anim_fname = '{hrrr_goes_eval_anim_fname}'" 

# # # --- write HRRR vs ACARS
# # shit2write = shit2write + '\n' + "acars_hrrr_eval_fnames = [ "
# # for f in acars_hrrr_eval_fnames: shit2write = shit2write + "'" + f + "',"
# # shit2write = shit2write[:-1] + ']'

# with open(fname, 'w') as f: f.write(f'{shit2write}\n')

# print('*** previous day shit saved ***')
# print(fname)
# print(shit2write)


# In[22]:


fname = OutputDir + f'previous_day_prod_{in_config_loc}_{prod_goes_yyyymmdd}'

# --- write GOES vs HRRR previous day anim
shit2write = f"hrrr_goes_eval_anim_fname = '{hrrr_goes_eval_anim_fname}'" 

# # --- write HRRR vs ACARS
# shit2write = shit2write + '\n' + "acars_hrrr_eval_fnames = [ "
# for f in acars_hrrr_eval_fnames: shit2write = shit2write + "'" + f + "',"
# shit2write = shit2write[:-1] + ']'

with open(fname, 'w') as f: f.write(f'{shit2write}\n')

print('*** previous day shit saved ***')
print(fname)
print(shit2write)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


# ! jupyter nbconvert AEROMMA_Auto_ForecastTemplate_v2_Download_PriorDay.ipynb --to script


# In[ ]:





# In[ ]:


#### cp ~/my_python/AEROMMA_Auto_ForecastTemplate_v2_Download_PriorDay.py AEROMMA_Auto_ForecastTemplate_v2_Download_PriorDay.py


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[268]:





# In[ ]:





# In[ ]:





# In[ ]:




