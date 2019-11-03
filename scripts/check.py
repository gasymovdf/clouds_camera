import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from astropy.table import Column
from astropy.table import QTable
from matplotlib.colors import LogNorm
from photutils import CircularAperture
from astropy.coordinates import SkyCoord, AltAz
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import EarthLocation


def astrometry(file, location, time, RAastro=180., DECastro=0., scale_low=0.1, scale_high=180., tweak_order=2):
    os.system('/usr/local/astrometry/bin/solve-field  --no-background-subtraction --resort --downsample 2 --no-verify --scale-low ' + str(scale_low) +
              ' --scale-high ' + str(scale_high) + ' --overwrite --tweak-order ' + str(tweak_order) + ' ' + file) # ' --ra ' + str(RAastro) + ' --dec ' + str(DECastro) + 

    wcs_file = format2_wcs(file)
    with fits.open(wcs_file, mode='update') as hdu:
            CR_altaz(hdu, location, time)
            hdu.flush() 


def format2_wcs(file):
    k = 0
    for i in range(len(file)-1, 0, -1):
        if file[i]=='.':
            k = i
            break
    return file.replace(file[k:], ".wcs")


def CR_altaz(hdu, location, time):
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
    image = Image.open(file)
    xsize, ysize = image.size
    rgb = image.split()
    data_r = np.array(rgb[0].getdata()).reshape(ysize, xsize)
    data_g = np.array(rgb[1].getdata()).reshape(ysize, xsize)
    data_b = np.array(rgb[2].getdata()).reshape(ysize, xsize)
    return data_r, data_g, data_b


def create_wcs(grid, location, time):
    for file in grid:
        with fits.open(file, mode='update') as hdu:
            header = hdu[0].header
            az, alt = header['CRVAL1_AZ'], header['CRVAL2_ALT']
            print(header['CRVAL1'], header['CRVAL2'])
            CR = SkyCoord(az*u.deg, alt*u.deg, obstime=time, location=location, frame = 'altaz')       
            header['CRVAL1'] = CR.transform_to('icrs').ra.degree
            header['CRVAL2'] = CR.transform_to('icrs').dec.degree
            print(header['CRVAL1'], header['CRVAL2'])
            hdu.flush() 


def BSC_open(file):
    BSCatalogue = pd.read_csv(file, sep='|')
    BSCatalogue = drop(BSCatalogue, [0, 6])
    BSCatalogue = QTable.from_pandas(BSCatalogue)
    names = {'ra         ':'ra', 'dec        ':'dec', 'vmag ':'vmag'}
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
    return BSCatalogue


def BSC_xy(BSCatalogue, wcs):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='x')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='y')
    for i in range(len(BSCatalogue)):
        BSCatalogue[i]['x'], BSCatalogue[i]['y'] = wcs.wcs_world2pix(
            BSCatalogue[i]['ra'], BSCatalogue[i]['dec'], 0)
        BSCatalogue[i]['x'] = round(BSCatalogue[i]['x'])
        BSCatalogue[i]['y'] = round(BSCatalogue[i]['y'])
    return BSCatalogue


def BSC_altaz(BSCatalogue, location, time):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='az')
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='alt')
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            ra, dec = BSCatalogue[i]['ra'], BSCatalogue[i]['dec']
            star = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs') 
            altaz_star = star.transform_to(AltAz(obstime=time, location=location))
            BSCatalogue[i]['az'] = altaz_star.az.degree
            BSCatalogue[i]['alt'] = altaz_star.alt.degree
    return BSCatalogue


def BSC_extinction(BSCatalogue):
    R = 6400
    H = 8
    A_V_zenit = 0.2

    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='vmag_atmosph')
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            sin = np.sin(BSCatalogue[i]['alt'] * np.pi/180.)
            D = 4*R**2 * sin**2 + 4*(2*R*H + H**2)
            L = (-2 * R * sin + np.sqrt(D))/2
            ext = A_V_zenit * L/H
            BSCatalogue[i]['vmag_atmosph'] = BSCatalogue[i]['vmag'] + ext
    return BSCatalogue


def BSC_observ_flux(BSCatalogue, data, STAR=10, CRIT_m=5.):
    a, b = len(data[0]), len(data)
    FLUX_0m = 0
    num = 0
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Observ_flux')

    for i in range(len(BSCatalogue)):
        x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
        m = BSCatalogue[i]['vmag']
        if (x >= 0 and x < a) and (y >= 0 and y < b) and (m < CRIT_m):
            STAR_data = Cutout2D(data, (x, y), (2*STAR + 1, 2*STAR + 1)).data
            mean_star, median_star, std_star = sigma_clipped_stats(
                STAR_data, sigma=5.0)
            STAR_data = STAR_data - median_star

            BSCatalogue[i]['Observ_flux'] = photometry(STAR_data)
            zero_point = BSCatalogue[i]['Observ_flux']/(2.512**m)
            FLUX_0m += zero_point
            num += 1
    return BSCatalogue, FLUX_0m/num


def BSC_theory_flux(BSCatalogue, FLUX_0m=1000):
    BSCatalogue.add_column(Column(np.zeros(len(BSCatalogue))), name='Flux')
    for i in range(len(BSCatalogue)):
        BSCatalogue[i]['Flux'] = 2.512**(-float(BSCatalogue[i]['vmag'])) * FLUX_0m
    return BSCatalogue


def make_map(BSCatalogue, data, outfile, CIRCLE_RADIUS=20., lw=1., OBS_TH_value=1.5):
    a, b = len(data[0]), len(data)
    plt.figure(figsize=(8, 13))
    plt.imshow(data, cmap='Greys', origin='lower', norm=LogNorm())
    for i in range(len(BSCatalogue)):
        if BSCatalogue[i]['Observ_flux']:
            x, y = BSCatalogue[i]['x'], BSCatalogue[i]['y']
            if BSCatalogue[i]['Observ_flux'] < BSCatalogue[i]['Flux'] / OBS_TH_value:
                color = 'red'
            else:
                color = 'green'
            aperture = CircularAperture((x, y), r=CIRCLE_RADIUS)
            aperture.plot(color=color, lw=lw)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def processed(file):
    if '.jpg' in file:
        red, green, blue = jpg2_fits(file) 
        data = green
    else:
        fits_file = file
        hdulist = fits.open(fits_file)
        data = hdulist[0].data[1]
    STAR_size = 15
    KGO = EarthLocation(lat='43d44m46s', lon='42d40m03s', height=2112*u.m)
    time = Time('2019-10-15T00:00:00.1', format='isot', scale='utc')

    grid = glob.glob('../data/*.wcs')
    if len(grid) == 0:
        astrometry(file, KGO, time, RAastro=330., DECastro=50., scale_low=40., scale_high=150., tweak_order=2)
        wcs_file = format2_wcs(file)
        w = wcs.WCS(wcs_file)
    else:
        create_wcs(grid, KGO, time)
        w = wcs.WCS(grid[0])

    BSC_file = "../data/BSC_clean.txt"
    BSCatalogue = BSC_open(BSC_file)
    BSCatalogue = BSC_xy(BSCatalogue, w)
    BSCatalogue, FLUX_0m = BSC_observ_flux(BSCatalogue, data, STAR=STAR_size, CRIT_m=5.)

    BSCatalogue = BSC_altaz(BSCatalogue, KGO, time)
    BSCatalogue = BSC_extinction(BSCatalogue)
    BSCatalogue = BSC_theory_flux(BSCatalogue, FLUX_0m=FLUX_0m)
    make_map(BSCatalogue, data, '../processed/result.jpg', CIRCLE_RADIUS=STAR_size, lw=2., OBS_TH_value=1.2)


file = '../data/Cas.jpg'
processed(file)