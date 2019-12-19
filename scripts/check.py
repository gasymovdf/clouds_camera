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
    if '.jpg' in file:
        red, green, blue = jpg2_fits(file)
        data = green
        if time is not None:
            return data, time
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
                return data, time
    return data


def bkg(data, SNR=5, box_size=30, filter_size=3):
    '''
    Применение медианного фильтра к изображению.
    '''
    sigma_clip = SigmaClip(sigma=SNR)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    #hdu = fits.PrimaryHDU()
    fits.writeto('background.fits', bkg.background, overwrite=True)
    return data - bkg.background


def jpg2_fits(file):
    # Получение данных из jpg
    image = Image.open(file)
    xsize, ysize = image.size
    rgb = image.split()
    data_r = np.array(rgb[0].getdata()).reshape(ysize, xsize)
    data_g = np.array(rgb[1].getdata()).reshape(ysize, xsize)
    data_b = np.array(rgb[2].getdata()).reshape(ysize, xsize)
    return data_r, data_g, data_b


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


def BSC_open(file):
    '''
    Открытие каталога Bright Stars. Поправка некоторых ошибок при чтении каталога,
    пересчёт (RA, DEC) в десятичный формат.
    '''
    BSCatalogue = pd.read_csv(file, sep='|')
    BSCatalogue = BSCatalogue.drop(BSCatalogue.columns[0], axis=1)
    BSCatalogue = BSCatalogue.drop(BSCatalogue.columns[-1], axis=1)
    BSCatalogue = QTable.from_pandas(BSCatalogue)
    for i in BSCatalogue.colnames:
        BSCatalogue[i].name = i.replace(' ', '')

    for i in BSCatalogue:
        ra = [float(j) for j in i['ra'].split()]
        dec = [float(j) for j in i['dec'].split()]

        i['ra'] = str(np.round((ra[0] + ra[1]/60 + ra[2]/(60*60))*15, 8))
        if dec[0] > 0:
            i['dec'] = str(np.round(dec[0] + dec[1]/60 + dec[2]/(60*60), 8))
        else:
            i['dec'] = str(np.round(dec[0] - dec[1]/60 - dec[2]/(60*60), 8))
    BSCatalogue['ra'] = BSCatalogue['ra'].astype(float)
    BSCatalogue['dec'] = BSCatalogue['dec'].astype(float)
    BSCatalogue['vmag'] = BSCatalogue['vmag'].astype(float)
    BSCatalogue.sort('vmag')
    return BSCatalogue


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


def make_map(BSCatalogue, file, data, outfile, lw=1.):
    '''
    Создание карты облачности с отметкой проходящих и непроходящих анализ звёзд.
    Анализ: если видимый поток меньше на 2*std от теоретического потока, то мы отмечаем "облако".
    Зелёный кружок -- звезда проходит тест, красный -- на месте звезды "облако".
    Для получения карты прозрачности на данный момент не хватает точности измерений нуль-пункта.
    '''
    a, b = len(data[0]), len(data)
    g, r = 0, 0
    fig = plt.figure()  # figsize=(13, 8)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='Greys', origin='lower', norm=LogNorm())
    # make_constellations()

    for i in BSCatalogue:
        x, y = i['x'], i['y']
        if i['Observ_flux'] < i['Flux'] - i['sigma_Flux']*2:
            color = 'red'
            r += 1
        else:
            color = 'green'
            g += 1

        if 'Alp' in i['alt_name']:
            ax.text(x + 30, y + 30, i['alt_name'])
        circle = plt.Circle((x, y), 20, color=color, lw=lw, fill=False)
        ax.add_artist(circle)

    ax.plot([], [], ' ', label='green/(red + green) = ' +
            str(np.round(g/(r + g), 3)))
    ax.legend(loc='best')
    ax.set_xlim(0, a)
    ax.set_ylim(b, 0)
    ax.set_title('Cloud map for ' + file)
    ax.legend()
    # fig.tight_layout()
    fig.savefig(outfile)
    plt.close()


def processed(file, outfile, calibrate=True):  # Основная процедура программы
    data, t = get_data(file, TIME)
    data = bkg(data, SNR=SNR, box_size=box_size,
               filter_size=filter_size)
    loc = EarthLocation(lat=lat, lon=lon, height=height*u.m)
    time = Time(t, format='isot', scale='utc')

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

    BSCatalogue = BSC_open(BSC_file)
    BSCatalogue = BSC_xy(BSCatalogue, wcs_file, data, min_res=res_rad, edge=edge)
    BSCatalogue = Stars_on_image(BSCatalogue, data, CRIT_m=max_mag)
    BSCatalogue = BSC_altaz(BSCatalogue, loc, time)
    BSCatalogue = BSC_extinction(BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)

    BSCatalogue = BSC_theory_flux(BSCatalogue, wcs_file)
    BSCatalogue = BSC_observ_flux(BSCatalogue, data, STAR_PHOT=STAR_PHOT)

    ascii.write(BSCatalogue, 'BSCatalogue.csv', format='csv', fast_writer=False)
    data = get_data(file)
    make_map(BSCatalogue, file, data, outfile, lw=2.)


BSC_file = '../data/catalogs/BSC_clean.txt'  # Расположение каталога

lat = '43d44m46s'  # Широта места наблюдения (в таком формате)
lon = '42d40m03s'  # Долгота места наблюдения (в таком формате)
height = 2112  # Выоста места наблюдения (метры)
TIME = '2019-09-24T17:18:23'  # Время UTC момента съёмки

center_size = 200  # Область центра снимка "без" дисторсии для astrometry
out_cntf = '../data/center_1.fits'  # название выходного файла с центром снимка
scl_l = 10.  # Минимальный размер снимка (deg)
scl_h = 100.  # Максимальный размер снимка (deg)
tw_o = 0  # Порядок дисторсии (учитывается плохо)
calibrate_m = 3.6  # Максимальная величина звёзд калибровки
edge = 400  # насколько теоретически далеко за границу могут "вылезать" звёзды

SNR = 5  # sigma для фильтра фона неба (pix)
box_size = 50  # Размер box для фильтра фона неба (pix)
filter_size = 10  # Размер mesh для фильтра фона неба (pix)

res_rad = 10  # Размер кружка поиска звезды
max_mag = 5.  # Предельная величина до которой проводится анали

STAR_PHOT = 30  # Размер апертуры фотометрии звезды (pix)
A_V_zenith = 0.2  # Поглощение в зените
Hatm = 8  # Высота однородной атмосферы (км)
