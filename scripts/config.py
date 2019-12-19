import check
import glob
import time

f = open('time.txt', 'w')
files = glob.glob('../data/*.fits*') # ['../data/150frames2.fits', '../data/150frames3_clouds.fits', '../data/150frames1.fits', '../data/15frames2.fits', '../data/150frames3.fits', '../data/150frames4_clouds.fits', '../data/15frames1.fits']
for i in range(len(files)):
    start_time = time.time()
    outfile = '../processed/result_' + str(i) + '.jpg'  # Расположение output картинки
    if 'center' not in files[i]:
        check.processed(files[i], outfile, calibrate=False)
        f.write('file: ' + files[i] + '; time = ' + str(time.time() - start_time) + ' секунд' + '\n')
# outfile = '../processed/result_.jpg'
# check.processed(files[2], outfile)
f.close()