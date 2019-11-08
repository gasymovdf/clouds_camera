import os
import glob
import numpy as np
import pandas as pd
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
from photutils import Background2D, MedianBackground
from astropy.coordinates import SkyCoord, AltAz
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import EarthLocation


def astrometry(file, location, time, RAastro=180., DECastro=0., scale_low=0.1, scale_high=180., tweak_order=2):
    os.system('/usr/local/astrometry/bin/solve-field  --no-background-subtraction --resort --downsample 2 --no-verify --scale-low ' + str(scale_low) +
              ' --scale-high ' + str(scale_high) + ' --overwrite --tweak-order ' + str(tweak_order) + ' ' + file)  # ' --ra ' + str(RAastro) + ' --dec ' + str(DECastro) +

    wcs_file = format2_wcs(file)
    with fits.open(wcs_file, mode='update') as hdu:
        CR_altaz(hdu, location, time)
        hdu.flush()


def get_data(file):
    if '.jpg' in file:
        red, green, blue = jpg2_fits(file)
        data = green
    else:
        fits_file = file
        hdulist = fits.open(fits_file)
        data = hdulist[0].data[1]
    return data


def bkg(data, SNR=5, box_size=30, filter_size=3):
    sigma_clip = SigmaClip(sigma=SNR)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return data - bkg.background


def CR_altaz(hdu, location, time):
    # Сохранение в header горизонтальных координат позиционного пикселя, они должны сохраняться со временем
    header = hdu[0].header
    ra, dec = header['CRVAL1'], header['CRVAL2']
    CR = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    altaz_CR = CR.transform_to(AltAz(obstime=time, location=location))
    header['CRVAL1_AZ'] = altaz_CR.az.degree
    header['CRVAL2_ALT'] = altaz_CR.alt.degree


def drop(BSCatalogue, nums):
    for i in nums:
        BSCatalogue = BSCatalogue.drop(BSCatalogue.columns[i], axis=1)
    return BSCatalogue


def rename(BSCatalogue, names):
    for i in names:
        BSCatalogue[i].name = names[i]
    return BSCatalogue


def photometry(STAR_data):
    return np.sum(STAR_data)


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
        hdu.flush()


def Calculate_aperture(BSCatalogue, min_rad=5, increase_per_100px=5):
    p = np.array(BSCatalogue['Distance_center'])
    radii = min_rad + increase_per_100px * p/100
    return radii


def subtract_star(data, x, y, STAR_x, STAR_y, radius, STAR_SIZE):
    # Процедура "вычитания звезды", а именно приравнивание потока в квадратике звезды нулю
    a, b = len(data[0]), len(data)
    for i in range(max(int(x - radius + STAR_x - STAR_SIZE), 0), min(int(x - radius + STAR_x + STAR_SIZE + 1), a)):
        for j in range(max(int(y - radius + STAR_y - STAR_SIZE), 0), min(int(y - radius + STAR_y + STAR_SIZE + 1), b)):
            data[j][i] = 0
    return data


def BSC_open(file):
    BSCatalogue = pd.read_csv(file, sep='|')
    BSCatalogue = drop(BSCatalogue, [0, 6])
    BSCatalogue = QTable.from_pandas(BSCatalogue)
    names = {'ra         ': 'ra', 'dec        ': 'dec', 'vmag ': 'vmag'}
    BSCatalogue = rename(BSCatalogue, names)

    for i in range(len(BSCatalogue)):
        ra = [float(i) for i in BSCatalogue[i]['ra'].split()]
        dec = [float(i) for i in BSCatalogue[i]['dec'].split()]

        BSCatalogue[i]['ra'] = (ra[0] + ra[1]/60 + ra[2]/(60*60))*15
        if dec[0] > 0:
            BSCatalogue[i]['dec'] = dec[0] + dec[1]/60 + dec[2]/(60*60)
        else:
            BSCatalogue[i]['dec'] = dec[0] - dec[1]/60 - dec[2]/(60*60)
    BSCatalogue['ra'] = BSCatalogue['ra'].astype(float)
    BSCatalogue['dec'] = BSCatalogue['dec'].astype(float)
    BSCatalogue['vmag'] = BSCatalogue['vmag'].astype(float)
    BSCatalogue.sort('vmag')
    return BSCatalogue


def BSC_xy(BSCatalogue, wcs, data):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='Distance_center')
    a, b = len(data[0]), len(data)

    for i in range(len(BSCatalogue)):
        BSCatalogue[i]['x'], BSCatalogue[i]['y'] = wcs.wcs_world2pix(
            BSCatalogue[i]['ra'], BSCatalogue[i]['dec'], 0)
        BSCatalogue[i]['x'] = round(BSCatalogue[i]['x'])
        BSCatalogue[i]['y'] = round(BSCatalogue[i]['y'])
        x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
        BSCatalogue[i]['Distance_center'] = np.sqrt(
            (x - a/2)**2 + (y - b/2)**2)
    return BSCatalogue


def BSC_altaz(BSCatalogue, location, time):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='az')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='alt')
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            ra, dec = BSCatalogue[i]['ra'], BSCatalogue[i]['dec']
            star = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
            altaz_star = star.transform_to(
                AltAz(obstime=time, location=location))
            BSCatalogue[i]['az'] = altaz_star.az.degree
            BSCatalogue[i]['alt'] = altaz_star.alt.degree
    return BSCatalogue


def BSC_extinction(BSCatalogue, ObsAlt=0, A_V_zenith=0.2, Hatm=8):
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='vmag_atmosph')
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            zenith_angle = 90 - BSCatalogue[i]['alt']
            airmass = pyasl.airmassSpherical(zenith_angle, ObsAlt, yatm=Hatm)
            ext = A_V_zenith * airmass
            BSCatalogue[i]['vmag_atmosph'] = BSCatalogue[i]['vmag'] + ext
    return BSCatalogue


def BSC_observ_flux(BSCatalogue, data_obs, STAR=15, STAR_PHOT=5., CRIT_m=5.):
    a, b = len(data_obs[0]), len(data_obs)
    FLUX_0m = 0
    num = 0
    BSCatalogue.add_column(
        Column(np.zeros(len(BSCatalogue))), name='Observ_flux')

    for i in range(len(BSCatalogue)):
        x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
        m = BSCatalogue[i]['vmag']
        if (x >= 0 and x < a) and (y >= 0 and y < b) and (m < CRIT_m):
            if STAR is not 15:
                radius = STAR[i]
            else:
                radius = STAR

            STARregion_data = Cutout2D(
                data_obs, (x, y), (2*radius + 1, 2*radius + 1)).data
            STAR_y, STAR_x = np.unravel_index(
                np.argmax(STARregion_data), STARregion_data.shape)
            STAR_data = Cutout2D(
                STARregion_data, (STAR_x, STAR_y), (2*STAR_PHOT + 1, 2*STAR_PHOT + 1)).data
            BSCatalogue[i]['Observ_flux'] = photometry(STAR_data)
            data_obs = subtract_star(
                data_obs, x, y, STAR_x, STAR_y, radius, STAR_PHOT)

            zero_point = BSCatalogue[i]['Observ_flux'] * (2.512**m)
            FLUX_0m += zero_point
            num += 1
    return BSCatalogue, FLUX_0m/num


def BSC_theory_flux(BSCatalogue, FLUX_0m=1000):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Flux')
    for i in range(len(BSCatalogue)):
        BSCatalogue[i]['Flux'] = 2.512**(-float(BSCatalogue[i]
                                                ['vmag_atmosph'])) * FLUX_0m
    return BSCatalogue


def make_map(BSCatalogue, data, outfile, circle_radii=15, lw=1., OBS_TH_value=1.5):
    a, b = len(data[0]), len(data)
    g, r = 0, 0
    plt.figure(figsize=(8, 13))
    plt.imshow(data, cmap='Greys', origin='lower', norm=LogNorm())
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
            if BSCatalogue[i]['Observ_flux'] < BSCatalogue[i]['Flux'] / OBS_TH_value:
                color = 'red'
                r += 1
            else:
                color = 'green'
                g += 1

            if 'Alp' in BSCatalogue[i]['alt_name  ']:
                plt.text(x + 30, y + 30, BSCatalogue[i]['alt_name  '])
            if circle_radii is not 15:
                CIRCLE_RADIUS = circle_radii[i]
            else:
                CIRCLE_RADIUS = circle_radii

            aperture = CircularAperture((x, y), r=CIRCLE_RADIUS)
            aperture.plot(color=color, lw=lw)
    plt.plot([], [], ' ', label='min star flux (Obs/Th) = ' +
             str(np.round(OBS_TH_value, 3)))
    plt.plot([], [], ' ', label='green/(red + green) = ' +
             str(np.round(g/(r + g), 3)))
    plt.legend(loc='best')
    plt.colorbar()
    plt.xlim(0, a)
    plt.ylim(0, b)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def processed(file):  # Основная процедура программы
    data = get_data(file)  # Получение данных из картинки (fits или jpg)
    data = bkg(data, SNR=SNR, box_size=box_size,
               filter_size=filter_size)  # Процедура учёта фона неба
    loc = EarthLocation(lat=lat, lon=lon, height=height*u.m)
    time = Time(TIME, format='isot', scale='utc')

    # Проверка на наличие файлов сетки (предполагается, что камера неподвижна и объектив стоит тот же)
    grid = glob.glob('../data/*.wcs')
    if len(grid) == 0:  # Сетки нет
        astrometry(file, loc, time, scale_low=scl_l, scale_high=scl_h,
                   tweak_order=tw_o)  # генерация сетки с помощью astrometry
        wcs_file = glob.glob('../data/*.wcs')[0]
        w = wcs.WCS(wcs_file)
    else:
        # исправление позиционного пикселя на новые (RA, DEC) в новый момент времени
        create_wcs(grid[0], loc, time)
        w = wcs.WCS(grid[0])

    # Открытие каталога (СТРОГО ТОТ ЖЕ ФАЙЛ, ИНАЧЕ ВСЁ СЛОМАЕТСЯ) в формате astropy table, поправка названий колонок, СОРТИРОВКА ПО Vmag для анализа от самых ярких к самым тусклым звёздам
    BSCatalogue = BSC_open(BSC_file)
    # Перевод координат всех звёзд в x, y снимка, нахождение расстояния от центра
    BSCatalogue = BSC_xy(BSCatalogue, w, data)
    # Нахождение радиуса поиска звезды
    Aperture_radii = Calculate_aperture(BSCatalogue, min_rad=min_rad, increase_per_100px=incr)
    # Определение наблюдаемого потока от звезды
    BSCatalogue, FLUX_0m = BSC_observ_flux(
        BSCatalogue, data, STAR=Aperture_radii, STAR_PHOT=STAR_PHOT, CRIT_m=max_mag)

    # Перевод координат всех звёзд в Alt, Az
    BSCatalogue = BSC_altaz(BSCatalogue, loc, time)
    # Получение поглощения в атмосфере в приближении сферической однородной атмосферы
    BSCatalogue = BSC_extinction(
        BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)
    # Определяемого теоретического потока с учётом поглощения в атмосфере
    BSCatalogue = BSC_theory_flux(BSCatalogue, FLUX_0m=FLUX_0m)
    # Запись полученного каталога, чтобы посмотреть на адекватность полученных значений
    ascii.write(BSCatalogue, 'BSCatalogue.csv',
                format='csv', fast_writer=False)
    # Ещё раз получение данных (необходимо из-за вычетов квадратиков звёзд, которые сохраняются в data)
    data = get_data(file)
    make_map(BSCatalogue, data, outfile, circle_radii=Aperture_radii,
             lw=2., OBS_TH_value=OBS_TH_value)  # Получение итоговой картинки


file = '../data/Cas.jpg'  # Расположение файла
BSC_file = '../data/BSC_clean.txt'  # Расположение каталога
outfile = '../processed/result.jpg'  # Расположение output картинки

lat = '43d44m46s'  # Широта места наблюдения (в таком формате)
lon = '42d40m03s'  # Долгота места наблюдения (в таком формате)
height = 2112  # Выоста места наблюдения (метры)
TIME = '2019-10-15T00:00:00.1'  # Время UTC момента съёмки

SNR = 5  # sigma для фильтра фона неба (pix)
box_size = 20  # Размер box для фильтра фона неба (pix)
filter_size = 5  # Размер mesh для фильтра фона неба (pix)

scl_l = 40.  # Минимальный размер снимка (deg)
scl_h = 150.  # Максимальный размер снимка (deg)
tw_o = 2  # Порядок дисторсии (учитывается плохо)

min_rad = 10  # Размер кружка поиска звезды в центре снимка (pix)
incr = 6  # Увеличение кружка поиска звезды на 100 пикселей (pix)
max_mag = 5.  # Предельная величина до которой проводится анализ

# Во сколько раз наблюдаемый поток звезды должен быть меньше теоретического, чтобы принять её за облако (красный кружок)
OBS_TH_value = 1.2
STAR_PHOT = 6  # Размер апертуры фотометрии звезды (pix)
A_V_zenith = 0.2  # Поглощение в зените
Hatm = 8  # Высота однородной атмосферы (км)

processed(file)
