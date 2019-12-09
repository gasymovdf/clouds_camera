import check
import glob

files = glob.glob('../data/*.fits*') # ['../data/150frames2.fits', '../data/150frames3_clouds.fits', '../data/150frames1.fits', '../data/15frames2.fits', '../data/150frames3.fits', '../data/150frames4_clouds.fits', '../data/15frames1.fits']
check.processed(files[2])