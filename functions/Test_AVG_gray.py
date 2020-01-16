import cv2 as cv
import numpy as np 
import csv

def BGR_to_GRAY(bgr_list):
    bgr_list = np.uint8([[[bgr_list[0],bgr_list[1],bgr_list[2]]]])

    gray_avg_Values = cv.cvtColor(bgr_list,cv.COLOR_BGR2GRAY)
    gray_avg_Values = np.squeeze(gray_avg_Values)

    return gray_avg_Values

#immagine bgr
def avg_center(img, percentage = 10 ,threshold_list = [0,0,0]):
    '''
    This function returns  a numpy array containing the average  bgr value of a certain percentage of pixels starting from the center of an image.
    '''

    bgr_list = []
    height,width,_ = img.shape

    centerX = (width // 2 ) #
    centerY = (height // 2)                                                                  
                                                            # XLB,YTB --->    #####  <---- XRB,YTB
                                                                                #   #
                                                                                #   #
                                                            # XLB,YBB ---->   #####  <---- XRB,YBB

    #RightBorder 
    XRB = centerX + ((width * percentage)//200)                    
    #LeftBorder
    XLB = centerX - ((width * percentage)//200)
    #TopBorder
    YTB = centerY + ((height * percentage)//200)
    #BottomBorder
    YBB = centerY - ((height * percentage)//200)

    #print(f"{height}, {width} xrb  {XRB} ---- xlb {XLB} ----- ytb {YTB} ----- ybb {YBB}") 

    for x in range(XLB,XRB):
        for y in range(YBB,YTB):
            bgr_list.append(img[y,x])

    numpy_bgr_array = np.array(bgr_list)
    average_value = np.average(numpy_bgr_array,axis=0)

    average_value = average_value.astype(int)

    return average_value,True 

def main():
    path = ""
    
    try:
        csvfile = open('values.csv', 'w')
        writer = csv.writer(csvfile)
        writer.writerow(("NOME","avg_H","avg_S","avg_V"))

        for i in range(116,456):
            path = "/media/matte/ALLEMANDI/Sample_Data/zz_astropi_1_photo_" + str(i) + ".jpg"
            img = cv.imread(path)
            average_value,_ = avg_center(img)
            average_value = BGR_to_GRAY(average_value)
            
            writer.writerow(((path[-13:-4]),average_value))

    except IOError:
        print("Errore file")
        exit(0)

    csvfile.close()
    

if __name__ == "__main__":
    main()
