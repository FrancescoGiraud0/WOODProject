import numpy as np
import cv2

def contrast_stretch(im):
    """
    Performs a simple contrast stretch of the given image, from 5-95%.
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calculateStatistic(matrix):
    NDVIGraduation = {
        "<0.1": 0,
        "0.1-0.2" : 0,
        "0.2-0.3" : 0,
        "0.3-0.4" : 0,
        "0.4-0.5" : 0,
        "0.5-0.6" : 0,
        "0.6-0.7" : 0,
        "0.7-0.8" : 0,
        "0.8-0.9" : 0,
        "0.9-1" : 0
    }

    for matrix_row in matrix:
        for value in matrix_row:
            number = float(value) / 255

            if number < 0.1:
                counter = NDVIGraduation["<0.1"]
                counter = counter + 1
                NDVIGraduation["<0.1"] = counter

            elif number < 0.2:
                counter = NDVIGraduation["0.1-0.2"]
                counter = counter + 1
                NDVIGraduation["0.1-0.2"] = counter

            elif number < 0.3:
                counter = NDVIGraduation["0.2-0.3"]
                counter = counter + 1
                NDVIGraduation["0.2-0.3"] = counter

            elif number < 0.4:
                counter = NDVIGraduation["0.3-0.4"]
                counter = counter + 1
                NDVIGraduation["0.3-0.4"] = counter

            elif number < 0.5:
                counter = NDVIGraduation["0.4-0.5"]
                counter = counter + 1
                NDVIGraduation["0.4-0.5"] = counter

            elif number < 0.6:
                counter = NDVIGraduation["0.5-0.6"]
                counter = counter + 1
                NDVIGraduation["0.5-0.6"] = counter

            elif number < 0.7:
                counter = NDVIGraduation["0.6-0.7"]
                counter = counter + 1
                NDVIGraduation["0.6-0.7"] = counter

            elif number < 0.8:
                counter = NDVIGraduation["0.7-0.8"]
                counter = counter + 1
                NDVIGraduation["0.7-0.8"] = counter

            elif number < 0.9:
                counter = NDVIGraduation["0.8-0.9"]
                counter = counter + 1
                NDVIGraduation["0.8-0.9"] = counter

            else:
                counter = NDVIGraduation["0.9-1"]
                counter = counter + 1
                NDVIGraduation["0.9-1"] = counter

    return NDVIGraduation
    
def calculateNDVI(path):
    image = cv2.imread(path)
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.01

    ndvi = (r.astype(float) - b) / bottom
    ndvi = contrast_stretch(ndvi)
    ndvi = ndvi.astype(np.uint8)
    
    return ndvi

def run():
    file = open("ndvi.txt", 'w')
    for counter1 in range(116,255):
        ndvi = calculateNDVI("Sample_Data\zz_astropi_1_photo_" + str(counter1) +".jpg")
        
        median = 0
        counter2 = 0
        for ndvirow in ndvi:
            for number in ndvirow:
                median = median + number
                counter2 = counter2 + 1
        if counter2 != 0:
            median = float(median) / counter2
            #print('\n'+ "Sample_Data\zz_astropi_1_photo_"+ str(counter1) +".jpg" + '\n\n' + str(ndvi) + '\n\n' + str(calculateStatistic(ndvi)) + '\n\nMedian = ' + str(median) + '\n\n-------------------------------------------------------------------\n')
            file.write('Sample_Data\zz_astropi_1_photo_'+ str(counter1) +'.jpg\n\n' + str(ndvi) + '\n\n' + str(calculateStatistic(ndvi)) + '\n\nMedian = ' + str(median) + '\n\n-------------------------------------------------------------------\n')
    file.close()

if __name__ == '__main__':
    run()