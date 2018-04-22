import cv2
import numpy as np 

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.circle(img,(x,y),1,(255,255,255),-1)
                cv2.line(img,(ix,iy),(x,y),(255,255,255),10) # draw line between former and present pixel
                ix=x # save former x coordinate
                iy=y # save former y coordinate
    
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(img,(x,y),1,(255,255,255),-1)
            cv2.line(img,(ix,iy),(x,y),(255,255,255),10) # draw line between former and present pixel
            ix=x # save former x coordinate
            iy=y # save former y coordinate
    
    return x,y




img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('number input')
cv2.setMouseCallback('number input',interactive_drawing)
while(1):
    cv2.imshow('number input',img)

    k=cv2.waitKey(1)&0xFF  # escape key
    if k==27:
        break

cv2.imwrite('input.png',img)
cv2.destroyAllWindows()
