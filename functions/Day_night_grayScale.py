#import cv2 as cv
#import numpy as np 

def day_or_night(path,percentage):
    MIN_SAVE = 85
    for i in range(116,456):
        #path = "/media/matte/ALLEMANDI/Sample_Data/zz_astropi_1_photo_" + str(i) + ".jpg"
        img = cv.imread(path)
        #--------

        bgr_list = []
        height,width,_ = img.shape

        centerX = (width // 2 ) 
        centerY = (height // 2)                                                                  

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

        average_value = average_value.astype(int)  #qui ho average-value

        #----------------
        #average_value = BGR_to_GRAY(average_value)
        bgr_list = np.uint8([[[bgr_list[0],bgr_list[1],bgr_list[2]]]])

        gray_avg_Values = cv.cvtColor(bgr_list,cv.COLOR_BGR2GRAY)
        gray_avg_Values = np.squeeze(gray_avg_Values)               #qui ho il valore di grigio 

        if gray_avg_Values >= MIN_SAVE:
            #save image
        else:
            pass

    