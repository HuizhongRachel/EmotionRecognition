import cv2

smilepic = cv2.imread('../Decoration/smile.png')

small = cv2.resize(smilepic, (0,0), fx=0.2, fy=0.2)
print(len(small[0]))
print(len(small))
while True:

	cv2.namedWindow("newpic")
	cv2.imshow("newpic",small) 
	key = cv2.waitKey(40)
        if key == 32:
        	break

cv2.destroyWindow("preview")
