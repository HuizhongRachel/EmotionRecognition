import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m
from numpy import array

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
pop up an image showing the mouth with a blue rectangle
"""
def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', img) 
    cv.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array

def vectorize_imgonly(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = array(filename)
    im = Image.fromarray(im)
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array

if __name__ == '__main__':
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "../data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "../data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    
    """
    open webcam and capture images
    """
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        vc.set(3,720)
        vc.set(4,480)

        rval, frame = vc.read()
    else:
        rval = False

    print "\npress space to take picture; press ctl + c to exit"

    count = 0

    while rval:
        # cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(40)       
        img = cv.fromarray(frame)
        mouth = m.findmouth(img)
        
        if mouth != 2: # did not return error
            mouthimg = crop(mouth)
            result = lr.predict(vectorize_imgonly(mouthimg))
            if result == 1:
                
                count += 1
                print "you are smiling :-) "
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces: 
                    if count <=10 :                                     
                        smilepic = cv2.imread('../Decoration/1.jpg')
                    elif count <=20 :
                        smilepic = cv2.imread('../Decoration/2.jpg')
                    elif count <=30 :
                        smilepic = cv2.imread('../Decoration/3.jpg')
                    elif count <=39 :
                        smilepic = cv2.imread('../Decoration/4.jpg')
                    elif count ==40 :
                        smilepic = cv2.imread('../Decoration/4.jpg')
                        count = 0

                    smilepic = cv2.resize(smilepic, (0,0), fx=0.7*h/587, fy=0.7*h/587)
                    pic_w=len(smilepic[0])
                    pic_h=len(smilepic)
                    a=x+w/2-pic_w/2
                    b=y-pic_h

                    if a>=0 and b>=0 :
                        rows,cols,channels = smilepic.shape
                        img_mask = np.array(img)
                        roi = img_mask[b:rows+b,a:cols+a]

                        img2gray = cv2.cvtColor(smilepic,cv2.COLOR_BGR2GRAY)
                        ret, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY_INV)
                        mask_inv = cv2.bitwise_not(mask)

                        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                        smilepic_fg = cv2.bitwise_and(smilepic, smilepic, mask=mask)
                        dst = cv2.add(img_bg, smilepic_fg)
                        img_mask[b:rows+b,a:cols+a] = dst

                        cv2.imshow("preview",img_mask)
                        if key == 32: # press space to save images
                            cv.SaveImage("webcam.jpg", cv.fromarray(img_mask))
                    else :
                        cv2.imshow("preview", frame)
                        print "exceed"

            else:
                print "you are not smiling! :-| "
                cv2.imshow("preview", frame)
                

        else:
            cv2.imshow("preview", frame)
            print "failed to detect face. Try hold your head straight and make sure there is only one face."
    
    cv2.destroyWindow("preview")




