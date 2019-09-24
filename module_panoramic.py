
import os
import threading
import time
import utils
import module_result
import numpy as np
from module_result import create_object
import module_camera
import cv2 as cv
import module_detection

flag_pause = False
param_delay = 0.2
show_flag = utils.get_config('show').lower()

lock = threading.Lock()

static_images = [None, None, None]
static_objects = [[], [], []]
static_images_old = [None, None, None]

moving_images = [None, None, None]
moving_objects = [[], [], []]

final_objects = [[], [], []]

model_dir = utils.get_config('model')['directory']
input_image_rows = utils.get_config('model')['input_image_rows']
input_image_cols = utils.get_config('model')['input_image_cols']

net = module_detection.NetThread(-1, 'panoramic_net_thread', model_dir)
net.begin()


def diff_frame(frames):
    delta = cv.absdiff(frames[-2], frames[-1])
    delta = cv.cvtColor(delta, cv.COLOR_BGR2GRAY)
    delta = cv.medianBlur(delta, 3)
    for k in range(2, len(frames)):
        delta_old = cv.absdiff(frames[-k-1], frames[-k])
        delta_old = cv.cvtColor(delta_old, cv.COLOR_BGR2GRAY)
        delta_old = cv.medianBlur(delta_old, 3)
        delta_max = cv.max(delta, delta_old)
        delta = delta_max - delta_old

    ret, delta = cv.threshold(delta, 10, 255, cv.THRESH_BINARY)
    
    # erode
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    delta = cv.erode(delta, kernel)
    
    # erode
    # t0 = time.time()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 25))
    delta = cv.dilate(delta, kernel)
    # t1 = time.time()
    # print(t1 - t0)

    # filter
    delta = cv.medianBlur(delta, 3)

    return delta



def detect_moving_objects(frames, time_stamp=0.0, channel=-1):
    print(os.path.basename(__file__), utils.get_function_name())
    t0 = cv.getTickCount()
    
    frame_diff = diff_frame(frames)
    contours = cv.findContours(frame_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    t1 = cv.getTickCount()

    #print('detect_moving_objects time elapsed: %08.3f ms' % (1000.0*(t1-t0)/cv.getTickFrequency()))
    
    objs = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        obj = create_object(class_name='unknown',
                            state='moving',
                            time_stamp=time_stamp,
                            channel=channel,
                            bbox=[x,y,x+w,y+h],
                            score=0.0)
        objs.append(obj)
    
    return objs
    

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
            print("score = ", score)
            if score > 0.6:
                print("score is good enough")
                obj = create_object(class_name=obj_id,
                                    bbox=[xmin,ymin,xmax,ymax],
                                    state='static',
                                    score=score)
                objs.append(obj)
    objs = objs_nms(objs)
    return objs


def filter_moving_objects(img, objs):
    print(os.path.basename(__file__), utils.get_function_name())

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    rois = list()
    objs_ret = []
    for obj in objs:
        roi = obj['bbox']
        
        # filter bboxes
        max_pixels = 160
        xmin, ymin, xmax, ymax = tuple(roi)
        roi_w = xmax - xmin
        roi_h = ymax - ymin
        
        if obj['class_name'] == 'person':
            objs_ret.append(obj)
        elif obj['class_name'] == 'bird': #and roi_w <= max_pixels and roi_h <= max_pixels: # and (xmin >= img_cols//3 or ymax <= img_rows*0.6):
            objs_ret.append(obj)

    return objs_ret


def confirm_moving_objects(img, objs):
    print(os.path.basename(__file__), utils.get_function_name())

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    rois = list()
    for obj in objs:
        roi = obj['bbox']
        within_exist_count = 0
        for roi_exist in rois:
            if roi[0] >= roi_exist[0] and roi[1] >= roi_exist[1] and roi[2] <= roi_exist[2] and roi[3] <= roi_exist[3]:
                within_exist_count += 1
        
        if within_exist_count is 0:
            center = (roi[0] + roi[2])/2, (roi[1] + roi[3])/2
            cols = roi[2] - roi[0]
            rows = roi[3] - roi[1]
            if cols < input_image_cols: cols = input_image_cols
            if rows < input_image_rows: rows = input_image_rows
            
            if cols < rows * input_image_cols/input_image_rows: cols = rows * input_image_cols/input_image_rows
            if rows < cols * input_image_rows/input_image_cols: rows = cols * input_image_rows/input_image_cols
            
            xmin = round(center[0] - cols/2)
            ymin = round(center[1] - rows/2)
            xmax = round(center[0] + cols/2)
            ymax = round(center[1] + rows/2)

            if xmin < 0: xmin, xmax = 0, xmax-xmin
            if ymin < 0: ymin, ymax = 0, ymax-ymin
            if xmax > img_cols: xmax = img_cols
            if ymax > img_rows: ymax = img_rows

            roi = (xmin, ymin, xmax, ymax)
            rois.append(roi)

    print('confirm_moving_objects: rois =\n', rois)
    objs = detect_objs_in_rois(img, rois)
    return objs


def merge_objects_temp(objs_static, objs_moving, objs_moving_confirmed):
    objs = objs_moving_confirmed.copy() # confirmed moving objects have high confidence

    return objs



def merge_objects(objs_static, objs_moving, objs_moving_confirmed):
    objs = objs_moving_confirmed.copy() # confirmed moving objects have high confidence

    # add objs in objs_static and not in objs_moving
    for obj in objs_static:
        overlap_count = 0
        for obj_temp in objs_moving+objs_moving_confirmed:
            bbox1 = obj['bbox']
            bbox2 = obj_temp['bbox']
            ratio = calc_overlap_ratio(bbox1, bbox2, mode='min')
            if ratio > 0.5:
                overlap_count += 1
        
        if overlap_count is 0:
            objs.append(obj)

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


# detect moving objects, and merge static and moving objects
def proc_panoramic_moving(name, param_delay):
    global static_images
    print(os.path.basename(__file__), utils.get_function_name())
    
    for k in [1,2]:
        while static_images_old[k] is None:
            print('waiting for static image ready')
            time.sleep(param_delay)

    while True:
        while flag_pause is True:
            time.sleep(param_delay)

        for k in [1,2]:
            time_stamp = time.time()
            with lock:
                img_static = static_images[k].copy()
                img_static_old = static_images_old[k].copy()
                objs_static = static_objects[k].copy()
            
            ret, img = module_camera.get_image(k)
            if ret:
                objs_moving = detect_moving_objects([img_static_old, img_static, img], time_stamp=time_stamp, channel=k)
                objs_moving_confirmed = confirm_moving_objects(img, objs_moving)
                objs_moving_confirmed = filter_moving_objects(img, objs_moving_confirmed)
                objs_final = merge_objects(objs_static, objs_moving, objs_moving_confirmed)
                objs_final = filter_moving_objects(img, objs_final)
                
                with lock:
                    moving_images[k] = img.copy()
                    moving_objects[k] = objs_moving_confirmed.copy()
                    final_objects[k] = objs_final.copy()

                with module_result.lock:
                    module_result.result['panoramic'] = objs_final.copy()
        
                if show_flag == 'yes' or show_flag == 'true':
                    win_name = 'module_panoramic channel-%d' % k
                    module_result.show_image(win_name, img, objs_final)

        # time.sleep(param_delay)


def proc_panoramic_static(name, param_delay):
    global static_images
    print(os.path.basename(__file__), utils.get_function_name())
    
    while True:
        while flag_pause is True:
            time.sleep(param_delay)

        for channel in [1,2]:
            ret, img = module_camera.get_image(channel)
            if ret:
                objs = detect_static_objects(img, channel)
                with lock:
                    static_images_old[channel] = static_images[channel]
                    static_images[channel] = img.copy()
                    static_objects[channel] = objs.copy()
                #show the static result of panoramic
                with module_result.lock:
                    module_result.result['panoramic'] = objs.copy
                if show_flag == 'yes' or show_flag == 'true':
                    win_name = 'module_panoramic channel-%d' % channel
                    module_result.show_image(win_name, img, objs)
                
        time.sleep(0.1)


def proc_panoramic(*args, **kwargs):
    thread_panoramic_static = utils.my_thread(proc_panoramic_static, 'proc_panoramic_static', param_delay=param_delay)
    thread_panoramic_static.start()
    
    #Donot care about moving birds for 1.0
    #thread_panoramic_moving = utils.my_thread(proc_panoramic_moving, 'proc_panoramic_moving', param_delay=param_delay)
    #thread_panoramic_moving.start()

    #thread_panoramic_moving.join()
    thread_panoramic_static.join()



if __name__ == "__main__":
    """
    print("_______________111111111111111111111________________________________")
    path='../input/test_pics/'   #要裁剪的图片所在的文件夹
    filename='w20190805084121403_1.jpg'    #要裁剪的图片名
    image = cv.imread(path+filename,1)
    t0 = time.time()
    bboxes = get_split_bbox(image, dst_rows_min=300, dst_cols_min=533)
    print("len(bboxes)", len(bboxes))
    objs = detect_objs_in_rois(image, bboxes)
    t1 = time.time()
    print("t1-t0=", t1-t0)
    win_name = 'module_panoramic channel'
    module_result.show_image(win_name, image, objs)
    """
    proc_panoramic()