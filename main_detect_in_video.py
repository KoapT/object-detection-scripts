
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
#video_source = '/home/x/Videos/192.168.1.108_IP PTZ Dome_main_20190724193230.dav'

video_source = '/home/houxueyuan/anti-bird/anti-bird/input/panoramic-camera/10mniao.dav'

model_dir = utils.get_config('model')['directory']
input_image_rows = utils.get_config('model')['input_image_rows']
input_image_cols = utils.get_config('model')['input_image_cols']


net = module_detection.NetThread(-1, 'panoramic_net_thread', model_dir)
net.begin()




# rect: (xmin, ymin, xmax, ymax)
# mode: bbox~(r1&r2)/(r1|r2), min~(r1&r2)/min(r1,r2), max~(r1&r2)/max(r1,r2)
def calc_overlap_ratio(r1, r2, mode='bbox'):
    if mode not in ['bbox', 'min', 'max']: mode = 'bbox'

    w_inner = min(r1[2],r2[2]) - max(r1[0],r2[0])
    h_inner = min(r1[3],r2[3]) - max(r1[1],r2[1])
    w_outer = max(r1[2],r2[2]) - min(r1[0],r2[0])
    h_outer = max(r1[3],r2[3]) - min(r1[1],r2[1])

    if w_inner < 0: w_inner = 0
    if h_inner < 0: h_inner = 0
    if w_outer < 0: w_outer = 0
    if h_outer < 0: h_outer = 0

    area_inner = w_inner * h_inner
    area1 = (r1[2]-r1[0]) * (r1[3]-r1[1])
    area2 = (r2[2]-r2[0]) * (r2[3]-r2[1])
    if mode is 'max':
        area_outer = max(area1, area2)
    elif mode is 'min':
        area_outer = min(area1, area2)
    else:
        area_outer = w_outer * h_outer
        
    area_outer += 1.0e-3 # make sure area_outer > 0
    ratio = area_inner / area_outer # make sure denominator > 0

    return ratio



def detect_objs_in_rois(img, rois):
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    objs = list()
    imgs_list = list()

    if len(rois) == 0:
        return []
    
    for roi in rois:
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = tuple(roi)
        img_roi = img[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        imgs_list.append(img_roi)
    
    rst_list = net.callNet(imgs_list)
    
    for k in range(len(rois)):
        roi = rois[k]
        rst = rst_list[k]
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = tuple(roi)
        for obj in rst:
            xmin, ymin, xmax, ymax = tuple(obj[0:4])
            xmin, xmax = xmin + roi_xmin, xmax + roi_xmin
            ymin, ymax = ymin + roi_ymin, ymax + roi_ymin
            
            # check if the obj in image
            if xmin < 0 or xmin >= img_cols: continue
            if xmax < 0 or xmax >= img_cols: continue
            if ymin < 0 or ymin >= img_rows: continue
            if ymax < 0 or ymax >= img_rows: continue

            obj_id = int(obj[4])

            score = obj[5]
            
            obj = create_object(class_name=obj_id,
                                bbox=[xmin,ymin,xmax,ymax],
                                state='static',
                                score=score)
            objs.append(obj)
    objs = objs_nms(objs)
    return objs



def get_split_bbox(image, dst_rows_min=300, dst_cols_min=300, overlap_ratio_min = 0.10):
    src_rows = image.shape[0]
    src_cols = image.shape[1]
    bboxes = []
    
    for level in range(8):
        dst_rows = dst_rows_min * 2**level
        dst_cols = dst_cols_min * 2**level

        if dst_rows >= src_rows or dst_cols >= src_cols:
            bboxes.append((0,0,src_cols,src_rows))
            break

        overlap_rows_min = overlap_ratio_min*dst_rows
        overlap_cols_min = overlap_ratio_min*dst_cols
        
        num_rows = np.ceil( (src_rows-overlap_rows_min) / dst_rows )
        num_cols = np.ceil( (src_cols-overlap_cols_min) / dst_cols )

        step_rows = int((src_rows-dst_rows) / num_rows)
        step_cols = int((src_cols-dst_cols) / num_cols)

        for start_row in range(0, src_rows-dst_rows + 1, step_rows):
            for start_col in range(0, src_cols-dst_cols + 1, step_cols):
                bbox = (start_col, start_row, start_col+dst_cols, start_row+dst_rows) # x, y, w, h
                bboxes.append(bbox)
                #if start_col + dst_cols > src_cols: break

            #if start_row + dst_rows > src_rows: break

    return bboxes




# NMS： Non-Maximum Suppression
def objs_nms(objs):
    objs_ret = []
    for k1 in range(len(objs)):
        obj = objs[k1].copy()
        better_count = 0
        for k2 in range(len(objs)):
            if k1 == k2: continue
            bbox1, score1, class1 = objs[k1]['bbox'], objs[k1]['score'], objs[k1]['class_name']
            bbox2, score2, class2 = objs[k2]['bbox'], objs[k2]['score'], objs[k2]['class_name']
            if class1 != class2: continue
            ratio = calc_overlap_ratio(bbox1, bbox2, mode='min')
            if ratio > 0.5 and score1 < score2:
                better_count += 1
        
        if better_count is 0: objs_ret.append(obj)
    return objs_ret



def detect_static_objects(img, channel=-1):
    print(os.path.basename(__file__), utils.get_function_name())
    
    rois = get_split_bbox(img, dst_rows_min=input_image_rows, dst_cols_min=input_image_cols)

    objs = detect_objs_in_rois(img, rois)
    
    return objs



if __name__ == "__main__":
    cap = cv.VideoCapture(video_source)
    
    if cap.isOpened():
        print('open video success')
        print('video source: %s' % video_source)
        fps = cap.get(cv.CAP_PROP_FPS)
        print('fps = %f' % fps)
    else:
        print('open video %s fail.' % video_source)
        exit(-1)
    
    # loop forever to provide service
    frame_index = 0
    FRAME_INDEX_START = 8400
    FRAME_INDEX_END   = 9200
    while True:
        ret, img = cap.read()
        
        if not ret:
            break

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if flip:
            img = cv.flip(img, -1)

        if frame_index < FRAME_INDEX_START:
            module_result.show_image('detect-result', img, [])
        elif frame_index > FRAME_INDEX_END:
            break
        else:
            rois = get_split_bbox(img, dst_rows_min=input_image_rows, dst_cols_min=input_image_cols)
            objs = detect_objs_in_rois(img, rois)
            module_result.show_image('detect-result', img, objs)
        
        frame_index += 1
        print('frame_index = %d， completed %06.2f%%.' % (frame_index, 100*frame_index/FRAME_INDEX_END))
        
    print('ret is False')
