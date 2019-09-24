import cv2
import numpy as np
import matplotlib.pyplot as plt


video_file = '/home/wt/Desktop/repositories/anti-bird/input/panoramic-camera/10mniao.dav'
 
cap = cv2.VideoCapture(video_file)

if cap.isOpened():
    print('open video %s success.' % video_file)
else:
    print('open video %s fail.' % video_file)
    exit(-1)

print('frames count = %f' % cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames fps = %f' % cap.get(cv2.CAP_PROP_FPS))

def get_next_frame(cap):
    ret, frame = cap.read()

    t0 = cv2.getTickCount()
    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, -1)

    t1 = cv2.getTickCount()
    time_elapsed = 1000*(t1-t0)/cv2.getTickFrequency()
    print('preprocessing time_elapsed = %f' % time_elapsed)
    return ret, frame



def diff_frame(frames):
    cv2.namedWindow('diff0', cv2.WINDOW_NORMAL)
    cv2.namedWindow('diff1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('diff_max', cv2.WINDOW_NORMAL)
    delta = cv2.absdiff(frames[-2], frames[-1])
    delta = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    delta = cv2.medianBlur(delta, 3)
    cv2.imshow('diff0', delta)
    for k in range(2, len(frames)):
        delta_old = cv2.absdiff(frames[-k-1], frames[-k])
        delta_old = cv2.cvtColor(delta_old, cv2.COLOR_BGR2GRAY)
        delta_old = cv2.medianBlur(delta_old, 3)
        delta_max = cv2.max(delta, delta_old)
        delta = delta_max - delta_old
        cv2.imshow('diff1', delta_old)
        cv2.imshow('diff_max', delta_max)

    ret, delta = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)
    
    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    delta = cv2.erode(delta, kernel)

    # filter
    delta = cv2.medianBlur(delta, 3)

    return delta



cv2.namedWindow('frame_old', cv2.WINDOW_NORMAL)
cv2.namedWindow('frame_new', cv2.WINDOW_NORMAL)
cv2.namedWindow('frame_diff', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)

frames = list()
for k in range(3):
    ret, frame = get_next_frame(cap)
    frames.append(frame)

while ret:
    t0 = cv2.getTickCount()
    frame_diff = diff_frame(frames)
    t1 = cv2.getTickCount()
    time_elapsed = 1000*(t1-t0)/cv2.getTickFrequency()
    frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)

    img_result = frames[-1].copy()
    contours, hierarchy = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img_result,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.drawContours(img_result, contours, -1, (0,0,255), 1)

    cv2.putText(frame_diff, 'pos: %d, time_elapsed: %f ms' % (frame_index, time_elapsed), (100,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('frame_old',frames[-2])
    cv2.imshow('frame_new',frames[-1])
    cv2.imshow('frame_diff',frame_diff)
    cv2.imshow('result',img_result)
    key = cv2.waitKey()
    if key == 27:
        break

    for k in range(2):
        frames[k] = frames[k+1]
    ret, frame = get_next_frame(cap)
    frames[-1] = frame


cap.release()
cv2.destroyAllWindows()#


