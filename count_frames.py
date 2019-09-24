
import os
import threading
import time
import utils
import module_result
import numpy as np
import cv2 as cv
import module_detection
from module_result import create_object

flip = False
video_source = '/home/x/Videos/192.168.1.108_IP PTZ Dome_main_20190724193230.dav'

if __name__ == "__main__":
    cap = cv.VideoCapture(video_source)
    
    if cap.isOpened():
        print('open video success')
        print('video source: %s' % video_source)
        fps = cap.get(cv.CAP_PROP_FPS)
        print('fps = %f' % fps)
        frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        print('frame_count = %f' % frame_count)
    else:
        print('open video %s fail.' % video_source)
        exit(-1)
    
    # loop forever to provide service
    frame_index = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if flip:
            img = cv.flip(img, -1)
        
        module_result.show_image('detect-result', img, [])
        frame_index += 1
        print('frame_index = %d' % frame_index)
            
    print('ret is False')
