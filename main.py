"""
WOOD project data collector
Author -> WOOD Mission Team
"""

# Import modules
# -------------------------------
import logging
import logzero
from logzero import logger
from sense_hat import SenseHat
import ephem
from picamera import PiCamera
from picamera.array import PiRGBArray
import datetime
from time import sleep
import os

import cv2 as cv
import numpy as np
# -------------------------------

# Setup
# -------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))

# Connect to the Sense Hat
sh = SenseHat()

# Set a logfile name
logzero.logfile(dir_path+"/logData.csv")

# Set a custom formatter
formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s');
logzero.formatter(formatter)

# Latest TLE data for ISS location
name = "ISS (ZARYA)"
l1 = "1 25544U 98067A   18030.93057008  .00011045  00000-0  17452-3 0  9997"
l2 = "2 25544  51.6392 342.9681 0002977  45.8872  32.8379 15.54020911 97174"
iss = ephem.readtle(name, l1, l2)

# Picamera resolution
cam_resolution = (3280,2464)

# Set up camera
cam = PiCamera()
# Set the max resolution
cam.resolution = cam_resolution
# Set raw capture
rawCapture = PiRGBArray(cam, size=cam_resolution)
#--------------------------------

# Functions
#--------------------------------
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

    cam.exif_tags['GPS.GPSLongitude'] = '%d/1,%d/1,%d/10' % (
        long_value[0], long_value[1], long_value[2]*10 )

    lat_value = [float(i) for i in str(iss.sublat).split(":")]

    if lat_value[0] < 0:
        lat_value[0] = abs(lat_value[0])
        cam.exif_tags['GPS.GPSLatitudeRef'] = "S"
    else:
        cam.exif_tags['GPS.GPSLatitudeRef'] = "N"

    cam.exif_tags['GPS.GPSLatitude'] = '%d/1,%d/1,%d/10' % (
        lat_value[0], lat_value[1], lat_value[2]*10 )
    
    return(str(lat_value), str(long_value))

# functions/avgColorValue.py
def avg_color_value():
    '''
    This function returns  a numpy array containing the average bgr value of a
    certain percentage of pixels starting from the center of an image.
    '''

    bgr_list = []
    height,width,_ = img.shape

    centerX = (width // 2 )
    centerY = (height // 2)

    # XLB,YTB --->    #####  <---- XRB,YTB
    #                 #   #
    #                 #   #
    # XLB,YBB ---->   #####  <---- XRB,YBB

    #RightBorder 
    XRB = centerX + ((width * percentage)//200)                    
    #LeftBorder
    XLB = centerX - ((width * percentage)//200)
    #TopBorder
    YTB = centerY + ((height * percentage)//200)
    #BottomBorder
    YBB = centerY - ((height * percentage)//200)

    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x])

    numpy_bgr_array = np.array(bgr_list)
    average_value = np.average(numpy_bgr_array,axis=0)
    average_value = average_value.astype(int)

    return average_value,True 

def read_data(photo_counter):
    # Read some data from the Sense Hat, rounded to 4 decimal places
    temperature = round(sh.get_temperature(),4)
    humidity = round(sh.get_humidity(),4)

    # get latitude and longitude
    lat, lon = get_latlon()

    # Save the data to the file
    logger.info("%s,%s,%s,%s,%s", photo_counter, lat, lon, humidity, temperature )
#--------------------------------

def run():
    # create a datetime variable to store the start time
    start_time = datetime.datetime.now()
    # create a datetime variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.datetime.now()
    # run a loop for 2 minutes
    photo_counter = 1

    while (now_time < start_time + datetime.timedelta(minutes=178)):
        try:
            # Function 
            read_data(photo_counter)

            time.sleep(0.1)

            # Take a pic 
            camera.capture(rawCapture, format="bgr")
            image = rawCapture.array
            
            # Use zfill to pad the integer value used in filename to 3 digits (e.g. 001, 002...)
            #cam.capture(dir_path+"/photo_" + str(photo_counter).zfill(3) + ".jpg")
            
            # update the current time
            now_time = datetime.datetime.now()

        except Exception as e:
            logger.error("An error occurred: " + str(e))

run()
