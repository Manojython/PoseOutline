import cv2
import numpy as np


pose = cv2.imread("pose1.jpg")


pose = cv2.cvtColor(pose, cv2.COLOR_BGR2YCR_CB)
# hsv = cv2.cvtColor(pose, cv2.COLOR_BGR2YUV)
y, cb, cr = cv2.split(pose)
#cv2.imshow("initial",y)
#cv2.imshow("initial1",cb)
#cv2.imshow("initial2",cr)

Z = np.float32(cr)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 48
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((cr.shape))
param = -1,-1
count = 0
coordinate=[]
# def mousePosition(event,x,y,flags,param):
#      global count
#      global coordinate
#      if event == cv2.EVENT_LBUTTONDOWN:
#          print('x = %d, y = %d'%(x, y))
#          count = count + 1
         
#          if(len(coordinate)==4):
#              coordinate.clear()
#          else:
#              coordinate.append(x)
#              coordinate.append(y)
#          if count %2 == 0 and count > 0 :

#              cropimg(coordinate)   


#cv2.imshow('res2',res2)

sobely = cv2.Sobel(res2,cv2.CV_8U,0,1,ksize=5)
counter = 0
#cv2.imshow('sobely',sobely)
print(sobely.shape)
height, width = sobely.shape
for x in range(0,width):
    for y in range(0,height):
        if(sobely[y,x]<255):
            sobely[y,x]= 0
        else:
            if(counter == 0):
                counter = counter+1
                
                print(y,x)


cv2.imshow("masked",sobely)
# cv2.setMouseCallback('masked',mousePosition,param)
# def cropimg(cord):
    
#     cropped = sobely[cord[1]:cord[3] , cord[0]:cord[2]]
#     cv2.imshow("check",cropped)
cv2.imwrite("sob.jpg",sobely)
newsob = cv2.imread("sob.jpg")
cv2.circle(newsob,(208,324), 10, (0,0,255), 2)
cv2.imshow("marked",newsob)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(np.unique(sobely, axis=0, return_counts = True))