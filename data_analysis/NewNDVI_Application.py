import cv2 as cv
import numpy as np
import time

CAM_RESOLUTION = (2592,1944)    # Camera resoultion
CAM_FRAMERATE  = 15             # Camera Framerate
DIFF_THRESHOLD = 0.3            # Minimun threshold of contrast (Forest-Desert, Forest-Cities, Forest-Soil)
PIXEL_THRESHOLD = 0.6           # Divide by 10, is the minimun threshold for the contrast 
SIZE_PERCENTAGE = 30            # Percentage area of the picture starting from the center to apply is_day function
MIN_GREY_COLOR_VALUE = 70       # Minimun color value to save the photo
ML_MIN_N_OF_SAMPLES = 50        # Minimun pictures number to start the machine learning algorithm
CYCLE_TIME = 7                  # Cycle time in seconds

def contrast_stretch(im):
    '''
    Performs a simple contrast stretch of the given image, from 5-100%.
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
    '''
    This function calculates the NDVI (Normalized Difference
    Vegetation Index) for each pixel of the photo and collect
    these values in "ndvi" numpy array.
    '''
    # Extract bgr values
    b, _, r = cv.split(image)
    bottom = (r.astype(float) + b.astype(float))
    # Change zeros of bottom array  
    # (to make sure to not divide by zero)
    bottom[bottom == 0] = 0.0000000000001

    # Calculate NDVI value of each pixel
    ndvi = (r.astype(float) - b) / bottom

    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)
    
    return ndvi

def calculate_statistics(ndvi_array, pixel_threshold=PIXEL_THRESHOLD, diff_threshold=DIFF_THRESHOLD):
    '''
    This function generate a dictionary counting the percentage of pixels for every
    NDVI graduations (keys of the dictionary) of a numpy array made of NDVI values
    (a value for every pixel).
    This function also computes the 'diff' value, it is the percentage of pixels with
    a low NDVI (less vegetation) near every high NDVI pixels (every pixel with more
    than pixel_threshold NDVI value).
    '''
    NDVIGraduation = {
        0 : 0, # <0.1
        1 : 0, # 0.1-0.2
        2 : 0, # 0.2-0.3
        3 : 0, # 0.3-0.4
        4 : 0, # 0.4-0.5
        5 : 0, # 0.5-0.6
        6 : 0, # 0.6-0.7
        7 : 0, # 0.7-0.8
        8 : 0, # 0.8-0.9
        9 : 0, # 0.9-1.0
        'diff' : 0 # Number of pixel with contrast (Forest-Desert, Forest-Cities, Forest-Soil)
    }

    shape = np.shape(ndvi_array)
    # Map values from 0-255 to 0-1
    temp = ndvi_array / 255.0
    # Calculate the number of pixels
    nofpixel = 1.0 * shape[0] * shape[1]
    
    for i, val in enumerate(np.histogram(temp[1:-1,1:-1], bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])[0]):
        NDVIGraduation[i] = val/nofpixel
     
    diff_10 = np.where((temp - np.roll(temp,shift=1,axis=0)) > diff_threshold,1,0)
    diff_11 = np.where((temp - np.roll(temp,shift=1,axis=1)) > diff_threshold,1,0)
    diff_array_thr = np.where( temp>pixel_threshold, diff_10+diff_11, 0)
    
    NDVIGraduation['diff'] = (diff_array_thr[1:-1,1:-1].sum())/nofpixel

    return NDVIGraduation

def run():

    for photo_counter in range(1,99): #woodsmissionteam_img_006.jpg

        img = cv.imread("woodsmissionteam_img_" + str(photo_counter).zfill(3) +".jpg")

        ndvi = calculateNDVI(img)

        cv.imwrite("woodsmissionteam_img_" + str(photo_counter).zfill(3) +"_NDVI.jpg",ndvi)

        print(f'Computing photo number {photo_counter}...')

if __name__ == '__main__':
    run()