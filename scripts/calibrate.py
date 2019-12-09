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
    xrange = np.arange(xlims[0],xlims[1],0.1)
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y)[0]
    ax.plot(xrange, k*xrange + b, '--k', lw=0.3)

def line_k(x1, y1, x2, y2):
    return (y1 - y2), (x2 - x1), (x1*y2 - x2*y1)

def r(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def xy2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2xy(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def residual(p, x, y, data=None, x0=1000, y0=500):
    v = p.valuesdict()
    R, phi = xy2pol(x-x0, y-y0)
    R_obs = v['a']*R**3 + v['b']*R**2 + v['c']*R + v['d']
    xt, yt = pol2xy(R_obs, phi)
    if data is None:
        return xt+x0, yt+y0

    data_x = data[0]-x0    
    data_y = data[1]-y0    
    R_data, phi_data = xy2pol(data_x, data_y)
    return R_obs - R_data

def astrometry(file, location, time, RAastro=180., DECastro=0., scale_low=0.1, scale_high=180., tweak_order=2):
    os.system('/usr/local/astrometry/bin/solve-field  --no-background-subtraction --resort --downsample 2 --no-verify --scale-low ' + str(scale_low) +
              ' --scale-high ' + str(scale_high) + ' --overwrite --tweak-order ' + str(tweak_order) + ' ' + file)  # ' --ra ' + str(RAastro) + ' --dec ' + str(DECastro) +

    wcs_file = file.replace(file.split('.')[-1], 'wcs')
    with fits.open(wcs_file, mode='update') as hdu:
        CR_altaz(hdu, location, time)
        hdu.flush()


def CR_altaz(hdu, location, time):
    # Сохранение в header горизонтальных координат позиционного пикселя, они должны сохраняться со временем
    header = hdu[0].header
    ra, dec = header['CRVAL1'], header['CRVAL2']
    CR = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    altaz_CR = CR.transform_to(AltAz(obstime=time, location=location))
    header['CRVAL1_AZ'] = altaz_CR.az.degree
    header['CRVAL2_ALT'] = altaz_CR.alt.degree


def center(data, x, y, star_size=5):
    a = 2*star_size + 1
    star = Cutout2D(data, (x, y), (a, a)).data
    mass_x, mass_y = 0, 0
    for i in range(a):
        for j in range(a):
            mass_x += star[i][j]*j
            mass_y += star[i][j]*i
    x0 = mass_x/np.sum(star)
    y0 = mass_y/np.sum(star)
    x_r = x - star_size + x0
    y_r = y - star_size + y0
    return x_r, y_r


def xy_th(BSCatalogue, wcs, data):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')

    for i in BSCatalogue:
        i['x'], i['y'] = wcs.wcs_world2pix(i['ra'], i['dec'], 0)
    return BSCatalogue


def xy_obs(BSCatalogue, data, star_size=7):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x_obs')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y_obs')
    for i in BSCatalogue:
        x, y = i['x'], i['y']
        a, b = len(data[0]), len(data)
        if i['alt_name']: name = i['alt_name']
        else: name = i['name']

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

        x_obs, y_obs = [int(i) for i in input().split()]
        plt.close()
        i['x_obs'], i['y_obs'] = center(data, x_obs, y_obs, star_size=star_size)    
        # data[max(int(i['y_obs'] - star_size), 0):min(int(i['y_obs'] + star_size), len(data[0])),
        #      max(int(i['x_obs'] - star_size), 0):min(int(i['x_obs'] + star_size), len(data))] = 0

    return BSCatalogue


def save_params(header, p_r, x0=1000, y0=500):
    v = p_r.params.valuesdict()
    for i in v:
        header[i] = v[i]
    header['X0'] = x0
    header['Y0'] = y0

def observe_center(data, x, y, data_x, data_y, min_rad=5, min_norm_resid=500):
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
    p_r = lmfit.Parameters()
    p_r.add_many(('a', 1.), ('b', 1.), ('c', 1.), ('d', 1.))
    x = np.array([BSCatalogue['x']])[0]
    y = np.array([BSCatalogue['y']])[0]
    data_x = np.array([BSCatalogue['x_obs']])[0]
    data_y = np.array([BSCatalogue['y_obs']])[0]
    x0, y0 = observe_center(data, x, y, data_x, data_y)

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
    FLUX_0m = []
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Observ_flux')

    for i in BSCatalogue:
        x, y = i['x_obs'], i['y_obs']
        m = i['vmag_atmosph']
        STAR_data = Cutout2D(data, (x, y), (2*STAR_PHOT + 1, 2*STAR_PHOT + 1)).data
        # plt.imshow(STAR_data)
        # plt.show()
        i['Observ_flux'] = check.photometry(STAR_data)
        FLUX_0m.append(i['Observ_flux'] * (2.512**m))

    FLUX_0m = np.array(FLUX_0m)
    with fits.open(wcs_file, mode='update') as hdul:
        header = hdul[0].header
        header['zeropoint'] = FLUX_0m.mean()
        header['sigma_zeropoint'] = FLUX_0m.std()
        hdul.flush()   

    return BSCatalogue


def process(data, file, BSC_file, loc, time, scl_l=0.1, scl_h=180., tw_o=2, CRIT_m=3.5, height=0, A_V_zenith=0.2, Hatm=8):
    astrometry(file, loc, time, scale_low=scl_l, scale_high=scl_h,
                   tweak_order=tw_o)  # генерация сетки с помощью astrometry
    wcs_file = glob.glob('../data/*.wcs')[0]
    w = wcs.WCS(wcs_file)
    BSCatalogue = ascii.read('BSCatalogue_calibrate.csv', format='csv', fast_reader=False)
    BSCatalogue = check.BSC_open(BSC_file)
    BSCatalogue = xy_th(BSCatalogue, w, data)
    BSCatalogue = check.Stars_on_image(BSCatalogue, data, CRIT_m=CRIT_m)
    BSCatalogue = xy_obs(BSCatalogue, data, star_size=8)
    find_dist(BSCatalogue, data, wcs_file)

    BSCatalogue = check.BSC_altaz(BSCatalogue, loc, time)
    BSCatalogue = check.BSC_extinction(BSCatalogue, ObsAlt=height, A_V_zenith=A_V_zenith, Hatm=Hatm)
    BSCatalogue = find_zeropoint(BSCatalogue, data, wcs_file, STAR_PHOT=7)
    ascii.write(BSCatalogue, 'BSCatalogue_calibrate.csv',
                format='csv', fast_writer=False)
