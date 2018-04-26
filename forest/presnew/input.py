import cv2
import numpy as np 
import math
import scipy as sp
import tensorflow as tf

from scipy import ndimage

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

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def get_x_by_image(folder,image,reverse=False):
    # read the image
    gray = cv2.imread(folder+"/"+image, 0)

    # rescale it
    if reverse:
        gray = cv2.resize(255 - gray, (28, 28))
    else:
        gray = cv2.resize(gray, (28, 28))
    # better black and white version
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)

    rows, cols = gray.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """
    flatten = gray.flatten() / 255.0
    return flatten

def get_y_by_digit(digit):
    arr = np.zeros((10))
    arr[digit] = 1
    return arr

def get_learning_batch(folder,reverse=False):
    batch_xs = []
    batch_ys = []
    for file in os.listdir(folder):
        if file.endswith(".png"):
            digit = file[-5:-4]
            y = get_y_by_digit(digit)
            x = get_x_by_image(folder,file,reverse=reverse)
            batch_xs.append(x)
            batch_ys.append(y)
    return batch_xs, batch_ys

# main


#img = np.zeros((512,512,3), np.uint8)
img = np.zeros((512,512), np.uint8)

cv2.namedWindow('number input')
cv2.setMouseCallback('number input',interactive_drawing)
while(1):
    cv2.imshow('number input',img)
 
    k=cv2.waitKey(1)&0xFF
    if k==ord('d'): #delete : undo image
        #img = np.zeros((512,512,3), np.uint8)
        img = np.zeros((512,512), np.uint8)
    elif k==ord('e'):     #enter : save image
        break

cv2.imwrite('input.ppm',img)

cv2.destroyAllWindows()

# begin image processing

# read the image

# rescale it
img = cv2.resize(img, (28, 28))
# better black and white version
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

while np.sum(img[0]) == 0:
    img = img[1:]

while np.sum(img[:,0]) == 0:
    img = np.delete(img,0,1)

while np.sum(img[-1]) == 0:
    img = img[:-1]

while np.sum(img[:,-1]) == 0:
    img = np.delete(img,-1,1)

rows,cols = img.shape

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    # first cols than rows
    img = cv2.resize(img, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    # first cols than rows
    img = cv2.resize(img, (cols, rows))

colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')

shiftx,shifty = getBestShift(img)
shifted = shift(img,shiftx,shifty)
img = shifted

# save the processed images
cv2.imwrite("processed.ppm", img)
"""
all images in the training set have an range from 0-1
and not from 0-255 so we divide our flatten images
(a one dimensional vector with our 784 pixels)
to use the same 0-1 based range
"""
img = img.flatten() / 255.0
cv2.imwrite("flattened.ppm", img)
