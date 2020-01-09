import logging
import logzero
from logzero import logger
from sense_hat import SenseHat
import ephem
from picamera import PiCamera
import datetime
from time import sleep
import random
import os

import shutil

src = "sample_data/"
dst = "capture/"
imageName = "zz_astropi_1_photo_"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Connect to the Sense Hat
#sh = SenseHat()

# Set a logfile name
logzero.logfile(dir_path+"/data01.csv")

# Set a custom formatter
formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s');
logzero.formatter(formatter)

# Latest TLE data for ISS location
name = "ISS (ZARYA)"
l1 = "1 25544U 98067A   18030.93057008  .00011045  00000-0  17452-3 0  9997"
l2 = "2 25544  51.6392 342.9681 0002977  45.8872  32.8379 15.54020911 97174"
iss = ephem.readtle(name, l1, l2)

# Set up camera
cam = PiCamera()
#cam.resolution = (1296,972)

# function to write lat/long to EXIF data for photographs
def get_latlon():
    """
    A function to write lat/long to EXIF data for photographs
    """
    iss.compute() # Get the lat/long values from ephem
    long_value = [float(i) for i in str(iss.sublong).split(":")]
    if long_value[0] < 0:
        long_value[0] = abs(long_value[0])
        cam.exif_tags['GPS.GPSLongitudeRef'] = "W"
    else:
        cam.exif_tags['GPS.GPSLongitudeRef'] = "E"
    cam.exif_tags['GPS.GPSLongitude'] = '%d/1,%d/1,%d/10' % (long_value[0], long_value[1], long_value[2]*10)
    lat_value = [float(i) for i in str(iss.sublat).split(":")]
    if lat_value[0] < 0:
        lat_value[0] = abs(lat_value[0])
        cam.exif_tags['GPS.GPSLatitudeRef'] = "S"
    else:
        cam.exif_tags['GPS.GPSLatitudeRef'] = "N"
    cam.exif_tags['GPS.GPSLatitude'] = '%d/1,%d/1,%d/10' % (lat_value[0], lat_value[1], lat_value[2]*10)
    return(str(lat_value), str(long_value))


# create a datetime variable to store the start time
start_time = datetime.datetime.now()
# create a datetime variable to store the current time
# (these will be almost the same at the start)
now_time = datetime.datetime.now()
# run a loop for 2 minutes
photo_counter = 1

cnt = 116

while (now_time < start_time + datetime.timedelta(minutes=178)):
    try:
        # Read some data from the Sense Hat, rounded to 4 decimal places
        temperature = round(sh.get_temperature(),4)
        humidity = round(sh.get_humidity(),4)


        # get latitude and longitude
        #lat, lon = get_latlon()
        # Save the data to the file
        logger.info("%s,%s,%s", photo_counter,humidity, temperature ) #add lat lon
        # use zfill to pad the integer value used in filename to 3 digits (e.g. 001, 002...)
        #cam.captureDumpy(dir_path+"/photo_"+ str(photo_counter).zfill(3)+".jpg")
        
        #COPYING PHOTO FROM INITIAL DIRECTORY TO DESTINATION
        if cnt < 455:
            shutil.copy2((src+ imageName + cnt), dst)
            photo_counter+=1
            cnt += 1
        else:
            break
        # update the current time
        now_time = datetime.datetime.now()
    except Exception as e:
        logger.error("An error occurred: " + str(e))
