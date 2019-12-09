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


def get_data(file, time=None):
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
    sigma_clip = SigmaClip(sigma=SNR)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
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
    with fits.open(file, mode='update') as hdu:
        header = hdu[0].header
        az, alt = header['CRVAL1_AZ'], header['CRVAL2_ALT']
        CR = SkyCoord(az*u.deg, alt*u.deg, obstime=time,
                      location=location, frame='altaz')
        header['CRVAL1'] = CR.transform_to('icrs').ra.degree
        header['CRVAL2'] = CR.transform_to('icrs').dec.degree
        print(header['CRVAL1'], header['CRVAL2'])
        hdu.flush()


def Calculate_aperture(BSCatalogue, min_rad=5, increase_per_100px=5):
    p = np.array(BSCatalogue['Distance_center'])
    radii = min_rad + increase_per_100px * p/100
    return radii


def photometry(STAR_data):
    return np.sum(STAR_data)


def BSC_open(file):
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


def BSC_xy(BSCatalogue, wcs_file, data):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')
    w = wcs.WCS(wcs_file)
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    for i in BSCatalogue:
        xt, yt = w.wcs_world2pix(i['ra'], i['dec'], 0)
        for j in range(2):
            par = ['A', 'B']
            p = lmfit.Parameters()
            for k in range(1, 7):
                p.add('a' + str(k), header[par[j] + str(k)])
            if j == 1: i['x'] = clt.residual(p, xt, yt)
            else: i['y'] = clt.residual(p, yt, xt)
        
    return BSCatalogue


def Stars_on_image(BSCatalogue, data_obs, CRIT_m=5.):
    a, b = len(data_obs[0]), len(data_obs)
    i = 0
    while i < len(BSCatalogue):
        x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y'] 
        m = BSCatalogue[i]['vmag']
        if math.isnan(x) or math.isnan(y) or (x < 0 or x >= a) or (y < 0 or y > b) or (m > CRIT_m):
            BSCatalogue.remove_row(i)
            i-=1
        i+=1
    return BSCatalogue


def BSC_altaz(BSCatalogue, location, time):
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
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='vmag_atmosph')
    for i in BSCatalogue:
        zenith_angle = 90 - i['alt']
        airmass = pyasl.airmassSpherical(zenith_angle, ObsAlt, yatm=Hatm)
        ext = A_V_zenith * airmass
        i['vmag_atmosph'] = i['vmag'] + ext
    return BSCatalogue


def BSC_observ_flux(BSCatalogue, data, STAR_PHOT=5.):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Observ_flux')

    for i in BSCatalogue:
        x, y = i['x'], i['y']
        STAR_data = Cutout2D(data, (x, y), (2*STAR_PHOT + 1, 2*STAR_PHOT + 1)).data
        # plt.imshow(STAR_data)
        # plt.show()
        i['Observ_flux'] = photometry(STAR_data)
    return BSCatalogue


def BSC_theory_flux(BSCatalogue, wcs_file):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Flux')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='sigma_Flux')
    hdu = fits.open(wcs_file)
    header = hdu[0].header

    for i in BSCatalogue:
        i['Flux'] = 2.512**(-float(i['vmag_atmosph'])) * header['zeropoint']
        i['sigma_Flux'] = 2.512**(-float(i['vmag_atmosph'])) * header['sigma_zeropoint']
    return BSCatalogue


def make_map(BSCatalogue, data, outfile, lw=1.):
    a, b = len(data[0]), len(data)
    g, r = 0, 0
    fig = plt.figure(figsize=(8, 13))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='Greys', origin='lower', norm=LogNorm())
    # make_constellations()

    for i in BSCatalogue:
        x, y = i['x'], i['y']
        if i['Observ_flux'] < i['Flux'] - i['sigma_Flux']*1.96:
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
    ax.legend()
    # fig.tight_layout()
    fig.savefig(outfile)
    plt.close()


def processed(file):  # Основная процедура программы
    data, t = get_data(file, TIME) 
    data = bkg(data, SNR=SNR, box_size=box_size,
               filter_size=filter_size) 
    loc = EarthLocation(lat=lat, lon=lon, height=height*u.m)
    time = Time(t, format='isot', scale='utc')

    grid = glob.glob('../data/*.wcs')
    if len(grid) == 0:  # Сетки нет
        clt.process(data, file, BSC_file, loc, time, scl_l=scl_l, scl_h=scl_h, tw_o=tw_o, CRIT_m=calibrate_m, height=height, A_V_zenith=A_V_zenith, Hatm=Hatm)
    else:
        print("Если вы хотите откалибровать камеру, напишите 1:")
        a = input()
        if a == '1':
            clt.process(data, file, BSC_file, loc, time, scl_l=scl_l, scl_h=scl_h, tw_o=tw_o, CRIT_m=calibrate_m, height=height, A_V_zenith=A_V_zenith, Hatm=Hatm)
        else:
            create_wcs(grid[0], loc, time)
    wcs_file = glob.glob('../data/*.wcs')[0]

    BSCatalogue = BSC_open(BSC_file)
    BSCatalogue = BSC_xy(BSCatalogue, wcs_file, data)
    BSCatalogue = Stars_on_image(BSCatalogue, data, CRIT_m=max_mag)
    BSCatalogue = BSC_altaz(BSCatalogue, loc, time)
    ascii.write(BSCatalogue, 'BSCatalogue.csv',
                format='csv', fast_writer=False)
    BSCatalogue = BSC_extinction(BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)

    BSCatalogue = BSC_observ_flux(BSCatalogue, data, STAR_PHOT=STAR_PHOT)
    BSCatalogue = BSC_theory_flux(BSCatalogue, wcs_file)
    
    ascii.write(BSCatalogue, 'BSCatalogue.csv',
                format='csv', fast_writer=False)
    data = get_data(file) 
    make_map(BSCatalogue, data, outfile, lw=2.)  


BSC_file = '../data/catalogs/BSC_clean.txt'  # Расположение каталога
outfile = '../processed/result.jpg'  # Расположение output картинки

lat = '43d44m46s'  # Широта места наблюдения (в таком формате)
lon = '42d40m03s'  # Долгота места наблюдения (в таком формате)
height = 2112  # Выоста места наблюдения (метры)
TIME = '2019-09-24T17:18:23'  # Время UTC момента съёмки

scl_l = 30.  # Минимальный размер снимка (deg)
scl_h = 100.  # Максимальный размер снимка (deg)
tw_o = 0  # Порядок дисторсии (учитывается плохо)
calibrate_m = 3.5 # Максимальная величина звёзд калибровки

SNR = 5  # sigma для фильтра фона неба (pix)
box_size = 20  # Размер box для фильтра фона неба (pix)
filter_size = 5  # Размер mesh для фильтра фона неба (pix)

min_rad = 10  # Размер кружка поиска звезды в центре снимка (pix)
incr = 6  # Увеличение кружка поиска звезды на 100 пикселей (pix)
max_mag = 5.  # Предельная величина до которой проводится анализ

STAR_PHOT = 6  # Размер апертуры фотометрии звезды (pix)
A_V_zenith = 0.2  # Поглощение в зените
Hatm = 8  # Высота однородной атмосферы (км)