import cv2
import numpy as np 

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode

    markerSize = 8

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.circle(img,(x,y),markerSize,(255,255,255),-1)
                cv2.line(img,(ix,iy),(x,y),(255,255,255),thickness=markerSize*2,lineType=10) # draw line between former and present pixel
                ix=x # save former x coordinate
                iy=y # save former y coordinate
    
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(img,(x,y),markerSize,(255,255,255),-1)
            cv2.line(img,(ix,iy),(x,y),(255,255,255),thickness=markerSize*2,lineType=10) # draw line between former and present pixel
            ix=x # save former x coordinate
            iy=y # save former y coordinate
    
    return x,y

img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('number input')
cv2.setMouseCallback('number input',interactive_drawing)
while(1):
    cv2.imshow('number input',img)
 
    k=cv2.waitKey(1)&0xFF
    if k==ord('d'): #delete : undo image
        img = np.zeros((512,512,3), np.uint8)
    elif k==ord('e'):     #enter : save image
        break

height, width = img.shape[:2]
max_height = 28
max_width = 28

# only shrink if img is bigger than required
if max_height < height or max_width < width:
    # get scaling factor
    scaling_factor = max_height / float(height)
    if max_width/float(width) < scaling_factor:
        scaling_factor = max_width / float(width)
    # resize image
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

img = cv2.imwrite('input.png',img)

cv2.destroyAllWindows()
