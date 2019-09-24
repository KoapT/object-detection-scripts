
import os
import threading
import time
import utils
import module_result
import module_camera
import module_panoramic
import numpy as np


state_tracking = 'tracking'
state_searching = 'searching'
state_idle = 'idle'
state = state_searching
param_delay = 0.2
flag_pause = False
preset_flag = False

win_name = 'module_ptz'
ptz_channel = 0

show_flag = utils.get_config('show').lower()
tracking_class = utils.get_config('tracking_class')

default_ptz_z = utils.get_config('searching_zoom')
pid_p = utils.get_config('pid_p')
pid_i = utils.get_config('pid_i')


def calc_objs_shift(objs, img, limit=0.99):
    if objs is None: return None,None
    if len(objs) == 0: return None,None
    if img is None: return None,None
    
    rows, cols = img.shape[0:2]
    
    dx_min, dy_min = cols/2, rows/2
    
    for obj in objs:
        if obj['class_name'] in tracking_class:
            dx = obj['center'][0] - cols / 2
            dy = obj['center'][1] - rows / 2
            if (dx_min**2.0 + dy_min**2.0)**0.5 > (dx**2.0+dy**2.0)**0.5:
                dx_min = dx
                dy_min = dy

    if abs(dx_min) >= limit * cols/2 or abs(dy_min) >= limit * rows/2:
        return None, None
    
    return dx_min, dy_min



def calc_rotate_angle(objs, img, ptz_p, ptz_t, ptz_z):
    dx, dy = calc_objs_shift(objs, img)
    if dx is None or dy is None:
        return 0,0

    angle_p, angle_t =  module_camera.get_angle(dx, dy, ptz_z, img)
    
    beta  = ptz_t * np.pi / 180.0
    alpha = angle_p * np.pi / 180.0
    
    #theta = 2 * ( np.tan(alpha/2.0) / max(np.cos(beta),0.01) )
    theta = 2 * np.arctan( np.tan(alpha/2.0) / max(np.cos(beta),0.01) )

    err_p = theta * 180/np.pi
    err_t = angle_t

    return err_p, err_t


# get small moving objects from module panoramic
def get_small_moving_objects():
    print(os.path.basename(__file__), utils.get_function_name())
    return []


def count_birds_from_panoramic():
    print(os.path.basename(__file__), utils.get_function_name())
    birds_num = 0
    with module_panoramic.lock:
        for ch in [1,2]:
            for obj in module_panoramic.final_objects[ch]:
                if obj['class_name'] in tracking_class:
                    birds_num += 1
    return birds_num


def get_suspected_objects_from_panoramic():
    get_small_moving_objects()
    print(os.path.basename(__file__), utils.get_function_name())
    return []


def get_panoramic_birds():
    print(os.path.basename(__file__), utils.get_function_name())
    return []


def track_moving_birds(objs, img):
    if objs is None: return
    if len(objs) <= 0: return
    print(os.path.basename(__file__), utils.get_function_name())
    
    ptz_ret, ptz_p, ptz_t, ptz_z = module_camera.get_ptz()
    
    err_p, err_t = calc_rotate_angle(objs, img, ptz_p, ptz_t, ptz_z)

    ptz_p = ptz_p + err_p
    ptz_t = ptz_t + err_t
    module_camera.set_ptz(ptz_p, ptz_t, default_ptz_z,
                          block=True, err_p_max=1.0, err_t_max=1.0)
    
    err_p = [0, 0]
    err_t = [0, 0]

    time_stamp = [time.time()] * 2
    
    no_obj_count = 0
    while flag_pause is not True and ptz_ret == 0:
        ret, img = module_camera.get_image(ptz_channel, flip=False)
        if ret:
            rows = img.shape[0]
            cols = img.shape[1]
            rois = module_panoramic.get_split_bbox(img, dst_rows_min=rows, dst_cols_min=cols)
            objs = module_panoramic.detect_objs_in_rois(img, rois)
            
            if len(objs) == 0:
                no_obj_count += 1
                if no_obj_count > 5:
                    print('no object detect, tracking over.')
                    return
            else:
                no_obj_count = 0
                err_p[1] = err_p[0]
                err_t[1] = err_t[0]
                err_p[0], err_t[0] = calc_rotate_angle(objs, img, ptz_p, ptz_t, ptz_z)

                time_stamp[1] = time_stamp[0]
                time_stamp[0] = time.time()
                time_span = time_stamp[0] - time_stamp[1]
                if time_span > 1.0: time_span = 1.0
                print('time_span: ', time_span)

                # ptz_p = ptz_p + time_span*(pid_i*err_p[0] + pid_p*(err_p[0]-err_p[1]))
                # ptz_t = ptz_t + time_span*(pid_i*err_t[0] + pid_p*(err_t[0]-err_t[1]))
                # module_camera.set_ptz(ptz_p, ptz_t, default_ptz_z, block=False)
                ptz_p = ptz_p + err_p[0]
                ptz_t = ptz_t + err_t[0]
                module_camera.set_ptz(ptz_p, ptz_t, default_ptz_z,
                                      block=True, err_p_max=0.2, err_t_max=0.2)
                time.sleep(0.05)
            
            if show_flag == 'yes' or show_flag == 'true':
                module_result.show_image(win_name, img, objs)

    if flag_pause:
        print('module_ptz is paused.')
    
    return


# search objects, share dnn model with panoramic
zoom_value = module_camera.ratio_pixel2angle[0] / module_camera.ratio_pixel2angle[default_ptz_z]
ptz_p_min, ptz_p_delta, ptz_p_max = 0, 45/zoom_value, 360
ptz_t_min, ptz_t_delta, ptz_t_max = 0, 30/zoom_value, 60
search_ptz_p, search_ptz_t = ptz_p_min, ptz_t_min
ptz_set_count = 0
def search_static_objects():
    global search_ptz_p
    global search_ptz_t
    global ptz_set_count
    print(os.path.basename(__file__), utils.get_function_name())

    while flag_pause is not True:
        #if count_birds_from_panoramic() > 0:
            #return [], None
        module_camera.set_ptz(search_ptz_p, search_ptz_t, default_ptz_z) # search_ptz_p, search_ptz_t, zoom
        ptz_set_count += 1

        search_ptz_p += ptz_p_delta
        if search_ptz_p > ptz_p_max:
            search_ptz_t += ptz_t_delta
            search_ptz_p = ptz_p_min
            if search_ptz_t > ptz_t_max: break

        time_stamp = time.time()
        ret, img = module_camera.get_image(ptz_channel, flip=False)

        if ret:
            rois = module_panoramic.get_split_bbox(img, dst_rows_min=1000, dst_cols_min=1000)
            objs = module_panoramic.detect_objs_in_rois(img, rois)
        
            if show_flag == 'yes' or show_flag == 'true':
                module_result.show_image(win_name, img, objs)

            dx, dy = calc_objs_shift(objs, img, limit=0.8)
            if dx is not None and dy is not None:
                return objs, img
    search_ptz_p, search_ptz_t = ptz_p_min, ptz_t_min # reset ptz state to default
    print('ptz_set_count = %d' % ptz_set_count)
    ptz_set_count = 0
    return [], None


def confirm_object(suspected_object):
    return [], None


#Edition 1.0:
#1.No tracking;
#2.Just detect static objects;
#3.Detect area: the given area from preset(p、t、z)
def proc_ptz(*args, **kwargs):
    while True:
        while flag_pause:
            time.sleep(param_delay)
        print(os.path.basename(__file__), utils.get_function_name())
        
        #panoramic_birds_num = count_birds_from_panoramic()
        #objs_suspected = get_suspected_objects_from_panoramic()
        #objs_confirmed = []
        
        objs = []
        img = None 
        #if not at preset,the ptz camara detect all area
        #if arrived the preset,get the preset infor and detect the specific area
        if preset_flag is not True:
            while True:
                objs, img = search_static_objects()
                if len(objs) <= 0: break
        else:
            while True:
                #get_config_from_preset_ini()   need to add
                #objs, img = search_preset_static_objects()   need to add
                if len(objs) <= 0: break
        """
        if panoramic_birds_num > 0:
            continue
        
        elif len(objs_suspected) > 0:
            objs_confirmed, img = confirm_object(objs_suspected)
        
        if len(objs_confirmed) > 0:
            track_moving_birds(objs_confirmed, img)
        else:
            while True:
                objs, img = search_static_objects()
                if len(objs) <= 0: break
                track_moving_birds(objs, img)
        """


if __name__ == "__main__":
    proc_ptz()