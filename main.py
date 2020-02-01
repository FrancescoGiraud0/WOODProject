'''
WOOD project data collector
Author: WOOD Mission Team
'''

# Import modules
# -------------------------------
import logging
import logzero
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

# Set a custom formatter for information log
info_formatter = logging.Formatter('%(name)s - %(asctime)-15s - %(levelname)s: %(message)s')
# Set a custom formatter for data log
data_formatter = logging.Formatter('%(name)s , %(asctime)-15s , %(message)s')
# Logger objects creation
info_logger    = logzero.setup_logger(name='info_logger', logfile=dir_path+'/data01.csv', formatter=info_formatter)
data_logger    = logzero.setup_logger(name='data_logger', logfile=dir_path+'/data02.csv', formatter=data_formatter)

# Latest TLE data for ISS location
name = 'ISS (ZARYA)'
l1   = '1 25544U 98067A   20016.35580316  .00000752  00000-0  21465-4 0  9996'
l2   = '2 25544  51.6452  24.6741 0004961 136.6310 355.9024 15.49566400208322'
iss  = ephem.readtle(name, l1, l2)

# Picamera resolution and framerate
CAM_RESOLUTION = (2592,1952) #(2592,1944)
CAM_FRAMERATE  = 32
DIFF_THRESHOLD = 0.3

# Set up camera
cam = PiCamera()
# Set the resolution
cam.resolution = CAM_RESOLUTION
# Set the framerate
cam.framerate = CAM_FRAMERATE

# Set rawCapture
rawCapture = PiRGBArray(cam, size = CAM_RESOLUTION)
#--------------------------------

# Functions
#--------------------------------
def contrast_stretch(im):
    '''
    This function performs a simple contrast stretch of the given image, from 5-100%.
    '''
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calculateNDVI(image):
    b, g, r = cv.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.0000000000001 # Make sure to not divide by zero

    ndvi = (r.astype(float) - b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)
    
    return ndvi

# function to write lat/long to EXIF data for photographs
def get_latlon():
    '''
    That function writes latitude and longitude to EXIF data
    and returns it.
    '''

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
    
    return str(lat_value), str(long_value)

def is_day(img, size_percentage=10, min_threshold=85):
    '''
    Function that return true if in the center size percentage of the photo,
    converted to gray color scale the average color value is more bright 
    than min_threshold (so, more simply, if it's day).
    '''
    bgr_list = []
    height,width,_ = img.shape

    centerX = (width // 2 )
    centerY = (height // 2)                                                                  

    #RightBorder 
    XRB = centerX + ((width * size_percentage)//200)                    
    #LeftBorder
    XLB = centerX - ((width * size_percentage)//200)
    #TopBorder
    YTB = centerY + ((height * size_percentage)//200)
    #BottomBorder
    YBB = centerY - ((height * size_percentage)//200)

    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x])

    numpy_bgr_array = np.array(bgr_list)
    average_value = np.average(numpy_bgr_array,axis=0)

    average_value = average_value.astype(int)

    average_value = np.uint8([[[average_value[0],average_value[1],average_value[2]]]])

    gray_avg_values = cv.cvtColor(average_value,cv.COLOR_BGR2GRAY)
    gray_avg_values = np.squeeze(gray_avg_values)

    return gray_avg_values >= min_threshold

def read_sh_data(take_pic, photo_id):
    if not take_pic:
        photo_id = str(None)

    # Read magnetometer data from the Sense Hat, rounded to 4 decimal places
    magnetometer_values = sh.get_compass_raw()
    magnetometer_values = map(lambda n: round(n,4), magnetometer_values)
    mag_x, mag_y, mag_z = magnetometer_values

    # Get latitude and longitude
    lat, lon = get_latlon()

    # Save the data to the log file
    data_logger.info("%s , %s , %s , %s , %s", photo_id, lat, lon, mag_x, mag_y, mag_z)
#--------------------------------

def run():
    # Creation of a datetime variable to store the start time
    start_time = datetime.datetime.now()
    # Creation of a datetime variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.datetime.now()
    
    # Counter to store the number of saved photos
    photo_counter = 1

    info_logger.info('Starting the experiment')

    # This will loop for 3 hours
    while (now_time < start_time + datetime.timedelta(minutes=5)):
        try:
            # Take a pic 
            cam.capture(rawCapture, format="bgr")
            image = rawCapture.array
            
            take_pic = is_day(image)

            info_logger.debug("Take pic: %s", take_pic)

            # Function that read all data from sense hat
            read_sh_data(take_pic, photo_counter)

            if take_pic:
                # Use zfill to pad the integer value used in filename to 3 digits (e.g. 001, 002...)
                file_name = dir_path + "/img_" + str(photo_counter).zfill(3) + ".jpg"

                # Saving the images
                cv.imwrite(file_name, image)

                info_logger.info('Photo %s saved', photo_counter )

                # Sleep for 5 seconds
                sleep(5)

                photo_counter += 1
            
                # It is necessary to take the next pic
                rawCapture.truncate(0)

            # Update the current time
            now_time = datetime.datetime.now()

        except Exception as e:
            info_logger.error("An error occurred: " + str(e))

        info_logger.info('End of the experiment')

run()