import cv2 as cv
import numpy as np 
#immagine bgr
def avg_center(percentage, path):
    '''
    This function returns  a numpy array containing the average  bgr value of a certain percentage of pixels starting from the center of an image.
    '''
    
    rgb_list = []

    img = cv.imread(path)
    height,width,_ = img.shape

    centerX = (width // 2 )
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
            rgb_list.append(img[y,x])
            #print((img[y,x]))

    numpy_bgr_array = np.array(rgb_list)
    average_value = np.average(numpy_bgr_array,axis=0)
    #print(f"valore mediio -- {average_value.astype(int)}")
    return average_value.astype(int)