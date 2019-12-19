#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import check
import glob
import time

f = open('time.txt', 'w')
files = glob.glob('../data/*.fits*')
for i in range(len(files)):
    start_time = time.time()
    outfile = '../processed/result_' + str(i) + '.jpg'  # Расположение output картинки
    if 'center' not in files[i]:
        check.processed(files[i], outfile, calibrate=False)
        f.write('file: ' + files[i] + '; time = ' + str(time.time() - start_time) + ' секунд' + '\n')
# outfile = '../processed/result_.jpg'
# check.processed(files[2], outfile)
f.close()