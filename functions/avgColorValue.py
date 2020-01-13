import cv2 as cv
import numpy as np 

def avg_color_value(img, percentage = 10 ,threshold_list = [0,0,0]):
    '''
    This function returns  a numpy array containing the average  bgr value of a certain percentage of pixels starting from the center of an image.
    '''

    bgr_list = []
    height,width,_ = img.shape

    centerX = width  // 2
    centerY = height // 2                                                     
                                                            
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