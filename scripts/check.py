#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
import glob
import math
import lmfit
import numpy as np
import pandas as pd
import calibrate as clt
from PIL import Image
from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.table import Column
from astropy.table import QTable
from PyAstronomy import pyasl
from matplotlib.colors import LogNorm
from photutils import CircularAperture
from astropy.stats import SigmaClip
import matplotlib.patches as mpatches
from photutils import Background2D, MedianBackground
from astropy.coordinates import SkyCoord, AltAz
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import EarthLocation
from astropy.coordinates import get_moon, get_sun
from astropy.utils import iers
import warnings
from astropy.utils.exceptions import AstropyWarning
iers.conf.auto_download = False
warnings.simplefilter('ignore', AstropyWarning)

def get_data(file, time=None):
    '''
    Получение данных с изображения (jpg или FITS). 
    Для FITS файла проводится нормировка потока, если есть FNUMBER и изменяется момент съёмки если есть TIME.
    '''
    camera_ip = 1
    if '.jpg' in file:
        red, green, blue = jpg2_fits(file)
        data = green
        if time is not None:
            return data, time, camera_ip
    else:
        fits_file = file
        hdulist = fits.open(fits_file)
        data = hdulist[0].data
        header = hdulist[0].header
        if header['FNUMBER']:
            data = data / int(header['FNUMBER'])
        if time is not None:
            if header['TIME']:
                time = header['TIME']
        return data, time, camera_ip

def bkg(data, SNR=5, box_size=30, filter_size=3):
    '''
    Применение медианного фильтра к изображению.
    '''
    sigma_clip = SigmaClip(sigma=SNR)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    #hdu = fits.PrimaryHDU()
    mean_bkg = np.mean(bkg.background)
    fits.writeto('background.fits', bkg.background, overwrite=True)
    return data - bkg.background, mean_bkg


def jpg2_fits(file):
    # Получение данных из jpg
    image = Image.open(file)
    xsize, ysize = image.size
    rgb = image.split()
    data_r = np.array(rgb[0].getdata()).reshape(ysize, xsize)
    data_g = np.array(rgb[1].getdata()).reshape(ysize, xsize)
    data_b = np.array(rgb[2].getdata()).reshape(ysize, xsize)
    return data_r, data_g, data_b


def WhereMoon(loc, time):
    moon = get_moon(time, loc)
    sun = get_sun(time)
    elongation = sun.separation(moon)
    ph_angle = np.arctan2(sun.distance*np.sin(elongation),
                      moon.distance - sun.distance*np.cos(elongation))
    ph = (1 + np.cos(ph_angle))/2.

    altaz_moon = moon.transform_to(AltAz(obstime=time, location=loc))
    altaz_sun = sun.transform_to(AltAz(obstime=time, location=loc))
    return ph, altaz_moon.az.deg, altaz_moon.alt.deg, altaz_sun.az.deg, altaz_sun.alt.deg


def create_wcs(file, location, time):
    '''
    Изменение координат (RA, DEC) позиционного пикселя на момент съёмки time.
    '''
    with fits.open(file, mode='update') as hdu:
        header = hdu[0].header
        az, alt = header['CRVAL1_AZ'], header['CRVAL2_ALT']
        CR = SkyCoord(az*u.deg, alt*u.deg, obstime=time,
                      location=location, frame='altaz')
        header['CRVAL1'] = CR.transform_to('icrs').ra.degree
        header['CRVAL2'] = CR.transform_to('icrs').dec.degree
        hdu.flush()


def photometry(STAR_data):
    '''
    Подобие фотометрии звезды, из-за плохих изображений звёзд находится просто сумма квадратика данных звезды. 
    '''
    return np.sum(STAR_data)

def bkg_mag(mean_bkg, wcs_file):
    fits_wcs = fits.open(wcs_file)
    header = fits_wcs[0].header
    pxl_wcs = wcs.WCS(header)
    scale = wcs.utils.proj_plane_pixel_scales(pxl_wcs)
    return -2.5*np.log10(mean_bkg/((scale[0]*3600)**2)/header['zeropoint'])


def BSC_xy(BSCatalogue, wcs_file, data, edge=400, min_res=5):
    '''
    Нахождение координат (x, y) в сетке снимка. 
    Используется слабый фильтр, чтобы убрать точно неправильные координаты.
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')
    a, b = len(data[0]), len(data)
    w = wcs.WCS(wcs_file)
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    for i in BSCatalogue:
        xt, yt = w.wcs_world2pix(i['ra'], i['dec'], 0)
        if np.isnan(xt) or np.isnan(yt) or (xt <= -edge or xt >= a+edge) or (yt <= -edge or yt >= b+edge):
            continue
        i['x'], i['y'] = clt.real_xy(xt, yt, header)
        # STAR = Cutout2D(data, (i['x'], i['y']), (2*min_res + 1, 2*min_res + 1)).data
        # i['y'], i['x'] = np.unravel_index(np.argmax(STAR), STAR.shape)
    return BSCatalogue


def Stars_on_image(BSCatalogue, data_obs, wcs=None, CRIT_m=5.):
    '''
    Фильтр звёзд, проходят только с (x, y) внутри изображения и m меньше заданной в главной части программы.
    Непрошедшие звёзды удаляются из каталога.
    '''
    a, b = len(data_obs[0]), len(data_obs)
    i = 0
    while i < len(BSCatalogue):
        x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
        m = BSCatalogue[i]['vmag']
        if math.isnan(x) or math.isnan(y) or (x <= 0 or x >= a) or (y <= 0 or y >= b) or (m > CRIT_m):
            BSCatalogue.remove_row(i)
            i -= 1
        else:
            if wcs is not None and BSCatalogue[i]['ch'] == 1:
                xt, yt = wcs.wcs_world2pix(
                    BSCatalogue[i]['ra'], BSCatalogue[i]['dec'], 0)
                BSCatalogue[i]['x'], BSCatalogue[i]['y'] = xt, yt
        i += 1
    return BSCatalogue


def BSC_altaz(BSCatalogue, location, time):
    '''
    Получение горизонтальныйх (Alt, Az) координат звёзд.
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='az')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='alt')
    for i in BSCatalogue:
        ra, dec = i['ra'], i['dec']
        star = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        altaz_star = star.transform_to(AltAz(obstime=time, location=location))
        i['az'] = altaz_star.az.degree
        i['alt'] = altaz_star.alt.degree
    return BSCatalogue


def BSC_extinction(BSCatalogue, ObsAlt=0, A_V_zenith=0.2, Hatm=8):
    '''
    Учёт поглощения в модели атмосферы в виде сферического слоя над Землёй. 
    '''
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='vmag_atmosph')
    for i in BSCatalogue:
        zenith_angle = 90 - i['alt']
        airmass = pyasl.airmassSpherical(zenith_angle, ObsAlt, yatm=Hatm)
        ext = A_V_zenith * airmass
        i['vmag_atmosph'] = i['vmag'] + ext
    return BSCatalogue


def BSC_observ_flux(BSCatalogue, data, STAR_PHOT=5.):
    '''
    Получение видимого потока от звёзд. Пересвеченные звёзды удаляются из каталога.
    '''
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='Observ_flux')
    j = 0
    while j < len(BSCatalogue):
        i = BSCatalogue[j]
        x, y = i['x'], i['y']
        STAR_data = Cutout2D(
            data, (x, y), (2*STAR_PHOT + 1, 2*STAR_PHOT + 1)).data
        ma = np.max(STAR_data)
        if ma > 200 and np.sum(STAR_data > 0.9*ma) > 12:
            BSCatalogue.remove_row(j)
            continue

        i['Observ_flux'] = photometry(STAR_data)
        # if i['Observ_flux'] < i['Flux'] - i['sigma_Flux']*2:
        # plt.imshow(STAR_data)
        # plt.title('r = ' + str(((x-960)**2 + (y-540)**2)**0.5) + '; FLUX = ' + str(i['Observ_flux']))
        # plt.show()
        j += 1

    return BSCatalogue


def BSC_theory_flux(BSCatalogue, wcs_file):
    '''
    Получение теоретического потока (и его погрешности) от звезды с учётом поглощения.
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Flux')
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='sigma_Flux')
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    for i in BSCatalogue:
        i['Flux'] = 2.512**(-float(i['vmag_atmosph'])) * header['zeropoint']
        i['sigma_Flux'] = 2.512**(-float(i['vmag_atmosph'])
                                  ) * header['sigma_zeropoint']
    return BSCatalogue

def make_constell(ax, data, wcs_file, HIP_file, cnstl_file):
    a, b = len(data[0]), len(data)
    w = wcs.WCS(wcs_file)
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    HIPcat = ascii.read(HIP_file, format='csv', fast_reader=True) 
    f = open(cnstl_file)
    cons_dict = {}
    for line in f:
        words = line.split()
        connects = [int(i) for i in words[2:]]
        cons_dict[words[0]] = connects
    for constell in cons_dict:
        cnct = cons_dict[constell]        
        for i in range(0, len(cnct), 2):
            ra_s1 = float(HIPcat[HIPcat['num'] == cnct[i]]['ra'])
            dec_s1 = float(HIPcat[HIPcat['num'] == cnct[i]]['dec'])
            ra_s2 = float(HIPcat[HIPcat['num'] == cnct[i+1]]['ra'])
            dec_s2 = float(HIPcat[HIPcat['num'] == cnct[i+1]]['dec'])
            xt1, yt1 = w.wcs_world2pix(ra_s1, dec_s1, 0)
            xt2, yt2 = w.wcs_world2pix(ra_s2, dec_s2, 0)
            if (np.isnan(xt1) or np.isnan(yt1) or (xt1 <= -edge or xt1 >= a+edge) or (yt1 <= -edge or yt1 >= b+edge)) and \
                (np.isnan(xt2) or np.isnan(yt2) or (xt2 <= -edge or xt2 >= a+edge) or (yt2 <= -edge or yt2 >= b+edge)):
                continue
            x1, y1 = clt.real_xy(xt1, yt1, header)
            x2, y2 = clt.real_xy(xt2, yt2, header)
            ax.plot([x1, x2], [y1, y2], lw=1, color='yellow')    


def make_eqgrid(ax, data, wcs_file):
    a, b = len(data[0]), len(data)
    w = wcs.WCS(wcs_file)
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    for dec in range(-90, 90, 20):
        dec_x = []
        dec_y = []
        for ra_i in range(0, 360, 1):
            xt, yt = w.wcs_world2pix(ra_i, dec, 0)
            if np.isnan(xt) or np.isnan(yt) or (xt <= -edge or xt >= a+edge) or (yt <= -edge or yt >= b+edge):
                continue
            x, y = clt.real_xy(xt, yt, header)
            dec_x.append(x)
            dec_y.append(y)
        ax.plot(dec_x, dec_y, color='blue', lw = 0.5)

    for ra in range(0, 360, 30):
        ra_x = []
        ra_y = []
        for dec_i in range(-90, 90, 1):
            xt, yt = w.wcs_world2pix(ra, dec_i, 0)
            if np.isnan(xt) or np.isnan(yt) or (xt <= -edge or xt >= a+edge) or (yt <= -edge or yt >= b+edge):
                continue
            x, y = clt.real_xy(xt, yt, header)
            ra_x.append(x)
            ra_y.append(y)
        ax.plot(ra_x, ra_y, color='blue', lw = 0.5)

def make_map(BSCatalogue, file, data, outfile, wcs_file, HIP_file, cnstl_file, lw=1.):
    '''
    Создание карты облачности с отметкой проходящих и непроходящих анализ звёзд.
    Анализ: если видимый поток меньше на 2*std от теоретического потока, то мы отмечаем "облако".
    Зелёный кружок -- звезда проходит тест, красный -- на месте звезды "облако".
    Для получения карты прозрачности на данный момент не хватает точности измерений нуль-пункта.
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Cloudiness')
    a, b = len(data[0]), len(data)
    g, r = 0, 0
    fig = plt.figure()  # figsize=(13, 8)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='Greys', origin='lower', norm=LogNorm())
    make_eqgrid(ax, data, wcs_file)
    make_constell(ax, data, wcs_file, HIP_file, cnstl_file)

    for i in BSCatalogue:
        x, y = i['x'], i['y']
        obs, th = i['Observ_flux'], i['Flux']
        i['Cloudiness'] = 1 - np.max((np.min((obs/th, 1)), 0))

    for i in BSCatalogue:
        x, y = i['x'], i['y']
        cl = i['Cloudiness']
        '''r = []
        for j in BSCatalogue:
            if i['name']!=j['name']:
                r.append(((x - j['x'])**2 + (y - j['y'])**2)**0.5)
        rad = np.min(r)'''
        if 'Alp' in i['alt_name']:
           ax.text(x + 30, y + 30, i['alt_name'])
        if cl > 0.4:
            circle = plt.Circle((x, y), 50, color=((255*cl)/255, (255*(1-cl))/255, 0), lw=lw, fill=True, alpha = 0.5)
            ax.add_artist(circle)

    cloud = np.mean(BSCatalogue['Cloudiness'])
    ax.set_xlim(0, a)
    ax.set_ylim(b, 0)
    ax.set_title('Cloud map for ' + file)
    # fig.tight_layout()
    fig.savefig(outfile)
    plt.close()
    '''
    cloud = plt.figure()  # figsize=(13, 8)
    ax_cloud = cloud.add_axes([0, 0, 1, 1])
    map = np.zeros((b, a))
    for i in range(0, b, 20):
        for j in range(0, a, 20):
            r = []
            for star in BSCatalogue:
                x, y = star['x'], star['y']
                r.append(((x - i)**2 + (y - j)**2)**0.5)
            nrst = np.argmin(r)
            obs, th = BSCatalogue['Observ_flux'][nrst], BSCatalogue['Flux'][nrst]
            map[i][j] = np.max((np.min((obs/th, 1)), 0))
    ax_cloud.imshow(map, cmap='magenta')
    ax.set_title('Cloud map for ' + file)
    # fig.tight_layout()
    fig.savefig('diff_map_' + outfile)
    plt.close()'''
    return cloud


def processed(file, outfile, calibrate=True):  # Основная процедура программы
    print("Working on: {}".format(file))
    data, t, camera_ip = get_data(file, TIME)
    data, mean_bkg = bkg(data, SNR=SNR, box_size=box_size,
               filter_size=filter_size)
    loc = EarthLocation(lat=lat, lon=lon, height=height*u.m)
    time = Time(t, format='isot', scale='utc')
    ph, moon_alt, moon_az, sun_alt, sun_az = WhereMoon(loc, time)

    grid = glob.glob('../data/*.wcs')
    if len(grid) == 0:  # Сетки нет
        clt.process(data, file, out_cntf, BSC_file, loc, time, center_size=center_size, scl_l=scl_l, scl_h=scl_h,
                    tw_o=tw_o, edge=edge, CRIT_m=calibrate_m, height=height, A_V_zenith=A_V_zenith, Hatm=Hatm, STAR_PHOT=STAR_PHOT)
    else:
        if calibrate:
            clt.process(data, file, out_cntf, BSC_file, loc, time, scl_l=scl_l, scl_h=scl_h, tw_o=tw_o,
                        edge=edge, CRIT_m=calibrate_m, height=height, A_V_zenith=A_V_zenith, Hatm=Hatm, STAR_PHOT=STAR_PHOT)
        else:
            create_wcs(grid[0], loc, time)
    wcs_file = glob.glob('../data/*.wcs')[0]
    bkg_magnitd = bkg_mag(mean_bkg, wcs_file)

    BSCatalogue = ascii.read(BSC_file, format='csv', fast_reader=True) 
    # BSCatalogue = BSC_open(BSC_file) # Работает на 5 секунд быстрее?
    BSCatalogue = BSC_xy(BSCatalogue, wcs_file, data, min_res=res_rad, edge=edge)
    BSCatalogue = Stars_on_image(BSCatalogue, data, CRIT_m=max_mag)
    BSCatalogue = BSC_altaz(BSCatalogue, loc, time)
    BSCatalogue = BSC_extinction(BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)

    BSCatalogue = BSC_theory_flux(BSCatalogue, wcs_file)
    BSCatalogue = BSC_observ_flux(BSCatalogue, data, STAR_PHOT=STAR_PHOT)
    
    ascii.write(BSCatalogue, 'BSCatalogue.csv', format='csv', fast_writer=False)
    data = get_data(file)
    cloud = make_map(BSCatalogue, file, data[0], outfile, wcs_file, HIP_file, cnstl_file, lw=2.)
    print("Output in: {}".format(outfile))
    return camera_ip, t, cloud, ph, moon_alt, moon_az, sun_alt, sun_az, bkg_magnitd


BSC_file = '../data/catalogs/BSCcat.csv'  # Расположение каталога
HIP_file = '../data/catalogs/HIPcat.csv'
cnstl_file = '../data/catalogs/constell.dat'

lat = '43d44m46s'  # Широта места наблюдения (в таком формате)
lon = '42d40m03s'  # Долгота места наблюдения (в таком формате)
height = 2112  # Высота места наблюдения (метры)
TIME = '2019-09-24T17:18:23'  # Время UTC момента съёмки

center_size = 200  # Область центра снимка "без" дисторсии для astrometry
out_cntf = '../data/center_1.fits'  # название выходного файла с центром снимка
scl_l = 10.  # Минимальный размер снимка (deg)
scl_h = 100.  # Максимальный размер снимка (deg)
tw_o = 0  # Порядок дисторсии (учитывается плохо)
calibrate_m = 3.6  # Максимальная величина звёзд калибровки
edge = 500  # насколько теоретически далеко за границу могут "вылезать" звёзды

SNR = 5  # sigma для фильтра фона неба (pix)
box_size = 50  # Размер box для фильтра фона неба (pix)
filter_size = 10  # Размер mesh для фильтра фона неба (pix)

res_rad = 10  # Размер кружка поиска звезды
max_mag = 5.  # Предельная величина до которой проводится анали

STAR_PHOT = 30  # Размер апертуры фотометрии звезды (pix)
A_V_zenith = 0.2  # Поглощение в зените
Hatm = 8  # Высота однородной атмосферы (км)
