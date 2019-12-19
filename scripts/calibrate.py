#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
import glob
import math
import check
import numpy as np
import pandas as pd
import lmfit
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

def drawLine2P(x,y,xlims,ax):
    '''
    Функция отрисовки прямой на изображении.
    '''
    xrange = np.arange(xlims[0],xlims[1],0.1)
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y)[0]
    ax.plot(xrange, k*xrange + b, '--k', lw=0.3)

def line_k(x1, y1, x2, y2):
    # Коэффициенты прямой, проходящей через 2 точки.
    return (y1 - y2), (x2 - x1), (x1*y2 - x2*y1)

def r(x1, y1, x2, y2):
    # Расстояние между двумя точками.
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def xy2pol(x, y):
    # Перевод прямоугольных (x, y) координат в полярные (rho, phi).
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2xy(rho, phi):
    # Перевод полярных (rho, phi) координат в прямоугольные (x, y) .
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def residual(p, x, y, data=None, x0=1000, y0=500):
    '''
    Функция для lmfit и по совместительству, возвращающая (x, y) с учётом дисторсии.
    Рассматривается модель радиальной дисторсии относительно некоторого центра.
    '''
    v = p.valuesdict()
    R, phi = xy2pol(x-x0, y-y0)
    R_obs = v['a']*R**4 + v['b']*R**3 + v['c']*R**2 + v['d']*R + v['e']
    xt, yt = pol2xy(R_obs, phi)
    if data is None:
        return xt+x0, yt+y0

    data_x = data[0]-x0    
    data_y = data[1]-y0    
    R_data, phi_data = xy2pol(data_x, data_y)
    return R_obs - R_data


def center_cutout(data, outfile, x0, y0, csize):
    # Вырезание центральной области изображения со слабой дисторсией для astrometry. 
    center = Cutout2D(data, (x0, y0), (2*csize + 1, 2*csize + 1)).data
    fits.writeto(outfile, center, overwrite=True)


def astrometry(cfile, location, time, x0=960, y0=540, c_size=250, RAastro=180., DECastro=0., scale_low=0.1, scale_high=180., tweak_order=2):
    '''
    Решение поля с помощью astrometry для получения плоской сетки координат.
    '''
    os.system('/usr/local/astrometry/bin/solve-field  --no-background-subtraction --downsample 2 --resort --no-verify --scale-low ' + str(scale_low) +
              ' --scale-high ' + str(scale_high) + ' --overwrite --tweak-order ' + str(tweak_order) + ' ' + cfile)  # ' --ra ' + str(RAastro) + ' --dec ' + str(DECastro)

    wcs_file = cfile.replace(cfile.split('.')[-1], 'wcs')
    with fits.open(wcs_file, mode='update') as hdu:
        CR_altaz(hdu, location, time, x0, y0, c_size)
        hdu.flush()


def CR_altaz(hdu, location, time, x0, y0, c_size):
    # Сохранение в header горизонтальных координат (Alt, Az) позиционного пикселя, они должны сохраняться со временем.
    header = hdu[0].header
    ra, dec = header['CRVAL1'], header['CRVAL2']
    CR = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    altaz_CR = CR.transform_to(AltAz(obstime=time, location=location))
    header['CRVAL1_AZ'] = altaz_CR.az.degree
    header['CRVAL2_ALT'] = altaz_CR.alt.degree
    header['CRPIX1'] = header['CRPIX1'] + x0 - c_size
    header['CRPIX2'] = header['CRPIX2'] + y0 - c_size 
    header['CTYPE1'] = 'RA---TAN'  
    header['CTYPE2'] = 'DEC--TAN'
    sip = ['A', 'B', 'AP', 'BP']
    for a in sip:
     del header[a + '_ORDER']
     for i in range(3):
      for j in range(3):
       if i+j <= 2:
         del header[a + '_' + str(i) + '_' + str(j)]


def center(data, x, y, star_size=5):
    '''
    Нахождение центроида звезды в выбранном квадрате.
    '''
    a = 2*(star_size) + 1
    star = Cutout2D(data, (x, y), (a, a)).data
    mass_x, mass_y = 0, 0
    for i in range(len(star)):
        for j in range(len(star[0])):
            mass_x += star[i][j]*j
            mass_y += star[i][j]*i
    x0 = mass_x/np.sum(star)
    y0 = mass_y/np.sum(star)
    x_r = x - min(x, star_size) + x0
    y_r = y - min(y, star_size) + y0
    print('x =', np.round(x_r, 3), ', y =', np.round(y_r, 3))
    return x_r, y_r

def real_xy(xt, yt, hdr):
    # Учёт дисторсии для теоретических "плоских" координат (xt, yt).
    x0, y0 = hdr['X0'], hdr['Y0']
    p = lmfit.Parameters()
    p.add_many(('a', hdr['A']), 
               ('b', hdr['B']),
               ('c', hdr['C']),
               ('d', hdr['D']),
               ('e', hdr['E']))
    return residual(p, xt, yt, x0=x0, y0=y0)


def xy_th(BSCatalogue, wcs, data, edge=250, hdr=None):
    '''
    Нахождение прямоугольных координат (x, y) на снимке для звёзд каталога без учёта дисторсии.
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='ch')
    a, b = len(data[0]), len(data)
    for i in BSCatalogue:
        if hdr is not None:
            xt, yt = wcs.wcs_world2pix(i['ra'], i['dec'], 0)
            obs = ascii.read('BSCatalogue_calibrate.csv', format='csv', fast_reader=False)
            if i['name'] in obs['name']:
                i['x'], i['y'] = xt, yt
            else:
                x, y = real_xy(xt, yt, hdr=hdr)
                i['x'], i['y'], i['ch'] = x, y, 1
        else:
            i['x'], i['y'] = wcs.wcs_world2pix(i['ra'], i['dec'], 0)   
    return BSCatalogue


def find_stars(data, x, y, name, star_size):
    '''
    Получение реальных (наблюдаемых) координат некоторых ярких звёзд от пользователя.
    В консоль выводится название звезды, её плоские координаты. 
    Также показывается карта, кругом отмечается возможная область располжения звезды и рисуется прямая вдоль которой идёт растяжение снимка.
    Если ввести (-1) звезда удалится из калибровочного каталога.
    '''
    a, b = len(data[0]), len(data)
    print('Укажите через пробел реальные координаты звезды {}(её примерные координаты: x={}, y={})'.format(name, round(x), round(y)))
    fig = plt.figure(figsize=(8, 13))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='Greys', origin='lower')
    circle = plt.Circle((x, y), 300, color='red', lw=1, fill=False)
    ax.add_artist(circle)
    drawLine2P([a/2, x], [b/2, y],[0, 3000], ax)
    plt.xlim(0, a)
    plt.ylim(0, b)
    plt.show()

    s = input()
    if s != '-1':
        x_obs, y_obs = [int(i) for i in s.split()]
        plt.close()
        return center(data, x_obs, y_obs, star_size=star_size)
    return -1, -1


def xy_obs(BSCatalogue, data, star_size=7, iter=0):
    '''
    Запись в каталог наблюдаемых координат звёзд.
    При первой итерации спрашиваются координаты всех подходящие звёзды из центральной области.
    При второй итерации спрашиваются координаты звёзд с края изображения (потому что для них есть первичный учёт дисторсии). 
    '''
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x_obs')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y_obs')
    j = 0
    while j < len(BSCatalogue):
        i = BSCatalogue[j]
        x, y = i['x'], i['y']
        if i['alt_name']: name = i['alt_name']
        else: name = i['name']
        if iter==0:
            i['x_obs'], i['y_obs'] = find_stars(data, x, y, name, star_size)
            if i['x_obs'] == -1 and i['y_obs'] == -1:
                BSCatalogue.remove_row(j)
                j-=1

        else:
            obs = ascii.read('BSCatalogue_calibrate.csv', format='csv', fast_reader=False)
            if i['name'] in obs['name']:
                i['x_obs'], i['y_obs'] = obs[i['name'] == obs['name']]['x_obs'],\
                                          obs[i['name'] == obs['name']]['y_obs']
            else:
                i['x_obs'], i['y_obs'] = find_stars(data, x, y, name, star_size)
                if i['x_obs'] == -1 and i['y_obs'] == -1:
                    BSCatalogue.remove_row(j)
                    j-=1
        j+=1
        # data[max(int(i['y_obs'] - star_size), 0):min(int(i['y_obs'] + star_size), len(data[0])),
        #      max(int(i['x_obs'] - star_size), 0):min(int(i['x_obs'] + star_size), len(data))] = 0

    return BSCatalogue


def save_params(header, p_r, x0=1000, y0=500):
    # Сохранение параметров полинома дисторсии и центра растяжения в header WCS сетки.
    v = p_r.params.valuesdict()
    for i in v:
        header[i] = v[i]
    header['X0'] = x0
    header['Y0'] = y0

def observe_center(data, x, y, data_x, data_y, min_rad=5, min_norm_resid=500):
    '''
    Нахождение центра дисторсии как медианы координат точек пересечений прямых (теор.-набл.) положения.
    Пересекается +/- в области центра снимка, но из-за погрешностей координат звёзд (из-за их изображений) 
    получаемые координаты далеко не очень хорошие. В данный момент функция не используется.
    '''
    a, b = len(data[0]), len(data)
    xi, yi = [], []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if r(x[i], y[i], data_x[i], data_y[i]) > min_rad and\
               r(x[j], y[j], data_x[j], data_y[j]) > min_rad:
                a1, b1, c1 = line_k(x[i], y[i], data_x[i], data_y[i])
                a2, b2, c2 = line_k(x[j], y[j], data_x[j], data_y[j])
                if a1/b1 != a2/b2:
                    x_interc = (b1/b2 * c2 - c1)/(a1 - b1/b2 * a2)
                    y_interc = (a1/a2 * c2 - c1)/(b1 - a1/a2 * b2)
                    if r(x_interc, y_interc, a/2, b/2) < min_norm_resid:
                        xi.append(x_interc)
                        yi.append(y_interc)

    if len(xi) > 0:
        xi = np.array(xi)
        yi = np.array(yi)
        return np.median(xi), np.median(yi)
    return a/2, b/2


def find_dist(BSCatalogue, data, wcs_file):
    '''
    Получение параметров дисторсии с помозью lmfit как полинома 4-й степени от расстояния до центра изображения.
    Нахождение центра дисторсии в данный момент убрано.
    Запись полученных параметров в header WCS файла.
    '''
    a, b = len(data[0]), len(data)
    p_r = lmfit.Parameters()
    p_r.add_many(('a', 1.), ('b', 1.), ('c', 1.), ('d', 1.), ('e', 1))
    x = np.array([BSCatalogue['x']])[0]
    y = np.array([BSCatalogue['y']])[0]
    data_x = np.array([BSCatalogue['x_obs']])[0]
    data_y = np.array([BSCatalogue['y_obs']])[0]
    x0, y0 = a/2, b/2 # observe_center(data, x, y, data_x, data_y)

    mi_r = lmfit.minimize(residual, params=p_r, nan_policy='omit', args=(x, y, [data_x, data_y], x0, y0))
    lmfit.printfuncs.report_fit(mi_r.params, min_correl=0.5)

    plt.plot(r(data_x, data_y, x0, y0), residual(mi_r.params, x, y, data=[data_x, data_y], x0=x0, y0=y0), '*')
    plt.title('residual (r_th - r_obs)')
    plt.xlabel('r_obs')
    plt.ylabel('r_th - r_obs')
    plt.savefig('residual.jpg')
    plt.close()
    
    with fits.open(wcs_file, mode='update') as hdul:
        header = hdul[0].header
        save_params(header, mi_r, x0, y0)
        hdul.flush()   


def find_zeropoint(BSCatalogue, data, wcs_file, STAR_PHOT=5):
    '''
    Нахождение интенсивности в отсчётах камеры для звезды 0-й звёздной величины.
    Пересвеченные звёзды не учитываются.
    Запись полученных параметров (нуль-пункт и его погрешность) в header WCS файла.
    '''
    FLUX_0m = []
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Observ_flux')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Flux_0m')

    for i in BSCatalogue:
        x, y = i['x_obs'], i['y_obs']
        m = i['vmag_atmosph']
        STAR_data = Cutout2D(data, (x, y), (2*STAR_PHOT + 1, 2*STAR_PHOT + 1)).data
        # plt.imshow(STAR_data)
        # plt.show()
        ma = np.max(STAR_data)
        if ma > 200 and np.sum(STAR_data > 0.9*ma) > 12:
            continue        

        i['Observ_flux'] = check.photometry(STAR_data)
        FLUX_0m.append(i['Observ_flux'] * (2.512**m))
        i['Flux_0m'] = FLUX_0m[-1]

    FLUX_0m = np.array(FLUX_0m)
    with fits.open(wcs_file, mode='update') as hdul:
        header = hdul[0].header
        header['zeropoint'] = FLUX_0m.mean()
        header['sigma_zeropoint'] = FLUX_0m.std()
        hdul.flush()   

    return BSCatalogue


def process(data, file, out_centerfile, BSC_file, loc, time, center_size=250, scl_l=0.1, scl_h=180., tw_o=2, edge=250, CRIT_m=3.5, height=0, A_V_zenith=0.2, Hatm=8, STAR_PHOT=15):
    a, b = len(data[0]), len(data)
    x0, y0 = a/2, b/2
    center_cutout(data, out_centerfile, x0, y0, center_size)
    astrometry(out_centerfile, loc, time, x0=x0, y0=y0, c_size=center_size, scale_low=scl_l, scale_high=scl_h,
                   tweak_order=tw_o)

    for i in range(0, 2):
        wcs_file = glob.glob('../data/*.wcs')[0]
        w = wcs.WCS(wcs_file)
        BSCatalogue = check.BSC_open(BSC_file)
        if i == 0:
            BSCatalogue = xy_th(BSCatalogue, w, data)
        else:
            hdu = fits.open(wcs_file)
            header = hdu[0].header
            BSCatalogue = xy_th(BSCatalogue, w, data, edge=edge, hdr=header)
        BSCatalogue = check.Stars_on_image(BSCatalogue, data, CRIT_m=CRIT_m, wcs=w)

        BSCatalogue = xy_obs(BSCatalogue, data, iter=i, star_size=STAR_PHOT)
        find_dist(BSCatalogue, data, wcs_file)

        BSCatalogue = check.BSC_altaz(BSCatalogue, loc, time)
        BSCatalogue = check.BSC_extinction(BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)
        BSCatalogue = find_zeropoint(BSCatalogue, data, wcs_file, STAR_PHOT=STAR_PHOT)
        ascii.write(BSCatalogue, 'BSCatalogue_calibrate.csv',
                    format='csv', fast_writer=False)
