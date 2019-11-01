from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Column
from astropy.table import QTable
from matplotlib.colors import LogNorm
from photutils import CircularAperture
from astropy.stats import sigma_clipped_stats
from astropy import wcs
from astropy.nddata.utils import Cutout2D
import os
import numpy as np


def astrometry(file, RAastro=180., DECastro=0., scale_low=0.1, scale_high=180., tweak_order=2):
    os.system('/usr/local/astrometry/bin/solve-field  --no-background-subtraction --resort --downsample 2 --no-verify --scale-low ' + str(scale_low) +
              ' --scale-high ' + str(scale_high) + ' --overwrite --ra ' + str(RAastro) + ' --dec ' + str(DECastro) + ' --tweak-order ' + str(tweak_order) + ' ' + file)

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
    fits_file = file
    hdulist = fits.open(fits_file)
    data = hdulist[0].data[1]
    header = hdulist[0].header[1]
    mean, median, std = sigma_clipped_stats(data, sigma=5.0)
    STAR_size = 15

    w = wcs.WCS('../data/new-image.wcs')
    BSC_file = "../data/BSC_clean.txt"
    BSCatalogue = BSC_open(BSC_file)
    BSCatalogue = BSC_xy(BSCatalogue, w)
    BSCatalogue, FLUX_0m = BSC_observ_flux(BSCatalogue, data, STAR=STAR_size, CRIT_m=5.)
    BSCatalogue = BSC_theory_flux(BSCatalogue, FLUX_0m=FLUX_0m)
    make_map(BSCatalogue, data, '../processed/result.jpg', CIRCLE_RADIUS=STAR_size, lw=2., OBS_TH_value=1.2)


file = '../data/cas_cus.fits'
processed(file)