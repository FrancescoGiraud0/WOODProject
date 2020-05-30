import cv2 as cv
import numpy as np
import time

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

def run():

    for photo_counter in range(1,100):

        img = cv.imread("woodsmissionteam_img_" + str(photo_counter).zfill(3) +".jpg")

        ndvi = calculateNDVI(img)

        ndvi = cv.cvtColor(ndvi, cv.COLOR_GRAY2RGB)

        cv.imwrite("woodsmissionteam_img_" + str(photo_counter).zfill(3) +"_NDVI.jpg",ndvi)

        ndvi_color = cv.applyColorMap(ndvi, cv.COLORMAP_JET)

        cv.imwrite("woodsmissionteam_img_" + str(photo_counter).zfill(3) +"_NDVI_COLOR_JET.jpg",ndvi_color)

        print(f'Computing photo number {photo_counter}...')

if __name__ == '__main__':
    run()