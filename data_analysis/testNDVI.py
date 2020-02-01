import numpy as np
import cv2 as cv
import time

DIFF_THRESHOLD = 0.3
PIXEL_THRESHOLD = 6

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def diff_calc(value, row, col, mat, threshold=0.3):
    diff=0

    if value - (mat[row-1][col]/255) > threshold:
        diff+=1
    
    if value - (mat[row+1][col]/255) > threshold:
        diff+=1

    if value - (mat[row][col-1]/255) > threshold:
        diff+=1
        
    if value - (mat[row][col+1]/255) > threshold:
        diff+=1

    return diff

def calculateStatistic(matrix, pixel_threshold, diff_threshold):
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
        9 : 0, # 0.9-1.00
        'diff' : 0 # Number of pixel with constrast (Forest-Desert, Forest-Cities)
    }

    for row in range(1,len(matrix)-1):
        for col in range(1,len(matrix[row])-1):
            number = float(matrix[row][col]) / 255
            key = np.floor(number * 10)

            if NDVIGraduation.get(key) != None:
                NDVIGraduation[key] += 1

            if key > pixel_threshold:
                NDVIGraduation['diff'] = diff_calc(number, row, col, matrix, threshold=diff_threshold)
            
    return NDVIGraduation
 
def calculateNDVI(path):
    image = cv.imread(path)
    b, g, r = cv.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.0000000000001 # Make sure to not divide by zero

    ndvi = (r.astype(float) - b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)
    
    return ndvi

def run():
    file = open("ndvi.txt", 'w')
    for photo_counter in range(116,255):
        ndvi = calculateNDVI("zz_astropi_1_photo_" + str(photo_counter) +".jpg")
        
        median = 0
        j = 0

        print(f'Computing photo number {photo_counter}...')

        for ndvirow in ndvi:
            for ndvi_value in ndvirow:
                median = median + ndvi_value
                j += 1

        if j != 0:
            median = float(median) / j
            print('Calculating statistics...')
            start_time = time.time()
            statistics_dict = calculateStatistic(ndvi, PIXEL_THRESHOLD, DIFF_THRESHOLD)
            print(f'Execution time {time.time()-start_time}')
            #print('\n'+ "Sample_Data\zz_astropi_1_photo_"+ str(counter1) +".jpg" + '\n\n' + str(ndvi) + '\n\n' + str(calculateStatistic(ndvi)) + '\n\nMedian = ' + str(median) + '\n\n-------------------------------------------------------------------\n')
            file.write('Sample_Data\zz_astropi_1_photo_'+ str(photo_counter) +'.jpg\n\n' +
                        str(ndvi) + '\n\n' + str(statistics_dict) + '\n\nMedian = ' + 
                        str(median) + '\n' + '-'*10)

    file.close()

if __name__ == '__main__':
    run()