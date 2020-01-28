#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import check
import glob
import time as programm_time
import psycopg2

conn = psycopg2.connect(user='jorji', password='12345', host='localhost')
cursor = conn.cursor()

f = open('time.txt', 'w')
files = glob.glob('../data/*.fits*')
for i in range(len(files)):
    start_time = programm_time.time()
    outfile = '../processed/result_' + str(i) + '.jpg'  # Расположение output картинки
    if 'center' not in files[i]:
        camera_ip, date_time, cloudiness, ph, moon_alt, moon_az, sun_alt, sun_az, bkg_mag = check.processed(files[i], outfile, calibrate=False)
        
        date = date_time.split('T')[0]
        time = date_time.split('T')[1] #.split(':')
        # time = time_array[0] + time_array[1] + time_array[2]
        cursor.execute("INSERT INTO test_table2 (camera, date, time, cloudiness, moon_phase, altitude_moon, azimuth_moon, altitude_sun, azimuth_sun, background_magnitude, raw_file, cloud_map) \
        	VALUES ({}, \'{}\', \'{}\', {}, {}, {}, {}, {}, {}, {}, \'{}\', \'{}\');".format(camera_ip, date, time, 1-cloudiness, ph, moon_alt, moon_az, sun_alt, sun_az, bkg_mag, files[i], outfile))
        conn.commit()
        f.write('file: ' + files[i] + '; time = ' + str(programm_time.time() - start_time) + ' секунд' + '\n')
# outfile = '../processed/result_.jpg'
# check.processed(files[2], outfile)
f.close()
cursor.close()
conn.close()