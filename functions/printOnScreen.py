def writeOnimage(img):
    font= cv.FONT_HERSHEY_SIMPLEX       #text font
    bottomLeftCornerOfText=(0,200)      #bottom Left Corner of the text
    fontScale=0.3                       #text scale
    fontColor=(255, 0, 0)               #textColor
    lineType= 1                         #line size
    text = 'Hello World!'               #text To Wrirte

    cv.putText(img,text,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)

    #Display the image
    #cv2.imshow("img",img)
