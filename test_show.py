import cv2
import numpy as np
import matplotlib.pyplot as plt 


#video_file = '/home/wt/Desktop/repositories/anti-bird/input/panoramic-camera/10mniao.dav'
video_file = '/home/houxueyuan/anti-bird/anti-bird/input/panoramic-camera/10mniao.dav'
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    print('open video %s success.' % video_file)
else:
    print('open video %s fail.' % video_file)
    exit(-1)

print('frames count = %f' % cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames fps = %f' % cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
ret,frame = cap.read()
print('ret =', ret)
while ret:
    cv2.imshow('frame',frame)
    cv2.waitKey(30)
    
    ret,frame = cap.read()
 
cap.release()
cv2.destroyAllWindows()#


