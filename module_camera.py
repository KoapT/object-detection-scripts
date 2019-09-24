
import os
import threading
import time
import utils
import module_result
import numpy as np
import cv2 as cv
import ctypes
from ctypes import pointer
so = ctypes.cdll.LoadLibrary
so_path = utils.get_config('so_path')
print('so_path:', so_path)
netdevlib = so(so_path)
print('netdevlib:', netdevlib)

camera_ip = "192.168.1.108"
camera_port = 37777
camera_status = netdevlib.DhInit(bytes(camera_ip,encoding="utf-8"), int(camera_port))

lock = threading.Lock()

video_sources = utils.get_config('video_sources')

video_channel_num = len(video_sources)

frames = [None] * video_channel_num


def get_ptz():
    if camera_status != 0:
        return -1,-1,-1,-1
    pan  = ctypes.c_int(0)
    tilt = ctypes.c_int(0)
    zoom = ctypes.c_int(0)
    result = netdevlib.DhQueryPtzInfo(ctypes.pointer(pan),
                                      ctypes.pointer(tilt),
                                      ctypes.pointer(zoom))
    pan, tilt, zoom = pan.value/10.0, tilt.value/10.0, zoom.value
    
    return result, pan, tilt, zoom


def set_ptz(ptz_p, ptz_t, ptz_z, block=True, timeout_secs=2.0,
            err_p_max = 0.1, err_t_max = 0.1, err_z_max = 1):
    if ptz_t < 1: ptz_t = 1
    if ptz_t > 90: ptz_t = 90

    err = netdevlib.DhPtzExactGoto(int(round(ptz_p*10)),
                                   int(round(ptz_t*10)),
                                   int(round(ptz_z)))
    
    t0 = time.time()
    while block:
        r, p, t, z = get_ptz()

        if r != 0:
            err = -3
            print('get_ptz error')
            break
        
        t1 = time.time()
        time_elapsed = t1 - t0

        err_p = abs(p - ptz_p)
        err_t = abs(t - ptz_t)
        err_z = abs(z - ptz_z)
        if err_p > 180: err_p -= 180
        if err_t > 180: err_t -= 180

        print('set_ptz: %6.3f secs, ' % time_elapsed
              + 'waiting ptz to reach %6.1f, %6.1f, %6.1f, ' % (ptz_p, ptz_t, ptz_z)
              + 'now ptz = %6.1f, %6.1f, %6.1f, ' % (p, t, z)
              + 'err ptz = %6.1f, %6.1f, %6.1f' % (err_p, err_t, err_z))

        if err_p <= err_p_max and err_t <= err_t_max and err_z <= err_z_max:
            print('ptz target reached.')
            break
        
        if time_elapsed > timeout_secs:
            err = -2
            break

    return err



ratio_pixel2angle =    [0.03114959,0.02971184,0.02381485,0.01999829,0.01803754,0.01569977,
                        0.01401799,0.01263166,0.01149508,0.01058291,0.01001924,0.00928891,
                        0.00870309,0.00810884,0.00764838,0.00721305,0.00694669,0.00658734,
                        0.00625223,0.00597793,0.00570268,0.00546307,0.00531205,0.00510159,
                        0.00491358,0.00472743,0.00456723,0.00440666,0.0043067 ,0.00416302,
                        0.00403231,0.00391183,0.00379269,0.00367863,0.00360951,0.00352154,
                        0.00343168,0.00333889,0.00325424,0.00317748,0.00313088,0.00305964,
                        0.00299512,0.00292324,0.00285087,0.00280464,0.00276532,0.00271303,
                        0.00265665,0.00261813,0.00257564,0.00252727,0.00251265,0.00246064,
                        0.00242179,0.00238719,0.00236334,0.00231687,0.0022931 ,0.0022643,
                        0.00224629,0.00220435,0.00218116,0.00215716,0.00212862,0.00211222,
                        0.00208632,0.00206715,0.00204596,0.00202769,0.00200704,0.00199502,
                        0.0019747 ,0.00195292,0.00194001,0.00192784,0.00190376,0.00189599,
                        0.00188276,0.00186846,0.00184404,0.00184121,0.00182984,0.00182278,
                        0.0017997 ,0.00180174,0.00178452,0.00177727,0.00177116,0.00176027,
                        0.001753  ,0.00174654,0.00174034,0.00171912,0.00171442,0.0017109,
                        0.00170265,0.0016961 ,0.00168502,0.0016746 ,0.00167397,0.00166888,
                        0.00166869,0.00165883,0.00165815,0.00164933,0.00164952,0.00164842,
                        0.00164816,0.00164461,0.00163916,0.00163985,0.00162909,0.00163018,
                        0.00163006,0.00163136,0.00163087,0.00161391,0.00162514,0.00162174,
                        0.00162132,0.00162971,0.00161411,0.00161224,0.00161487,0.00161198,
                        0.00160471,0.0015886]
def get_angle(dx, dy, zoom, img):
    zoom = round(zoom)
    if zoom < 0: zoom = 0
    if zoom >= len(ratio_pixel2angle): zoom = len(ratio_pixel2angle) - 1
    if img is None: return 0,0

    main_rows, main_cols = 1080, 1920
    rows, cols = img.shape[0:2]

    if rows <= 0 or cols <= 0: return 0,0

    ratio = ratio_pixel2angle[zoom]

    ptz_p =  ratio * dx/cols * main_cols
    ptz_t = -ratio * dy/rows * main_rows

    return ptz_p, ptz_t


def proc_capture_video(*args, **kwargs):
    video_channel = args[0]
    video_source = video_sources[video_channel]
    cap = cv.VideoCapture(video_source)
    
    if cap.isOpened():
        print('open video success')
        print('video_channel = %d' % video_channel)
        print('video source: %s' % video_source)
        fps = cap.get(cv.CAP_PROP_FPS)
        print('fps = %f' % fps)
    else:
        print('open video %s fail.' % video_source)
        exit(-1)
    
    # loop forever to privice service
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                frames[video_channel] = frame
        else:
            print('capture failed, video source: %s' % video_source)
            cap = cv.VideoCapture(video_source)
            if cap.isOpened():
                print('re-open video success')
                print('video_channel = %d' % video_channel)
                print('video source: %s' % video_source)
                fps = cap.get(cv.CAP_PROP_FPS)
                print('fps = %f' % fps)
            else:
                print('re-open video %s fail.' % video_source)
                exit(-1)
            continue
        


def get_image(video_channel=0, flip=True):
    with lock:
        if frames[video_channel] is not None:
            ret = True
            frame = frames[video_channel]
        else:
            ret = False
            frame = None

    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if flip:
            frame = cv.flip(frame, -1)
        
    return ret, frame


thread_capture_videos = []
for ch in range(video_channel_num):
    thread_capture_videos.append(utils.my_thread(proc_capture_video, ch))
    thread_capture_videos[ch].start()


for ch in range(video_channel_num):
    while get_image(video_channel=ch, flip=False)[0] is not True:
        print('waiting for video channel %d.' % ch)
        time.sleep(0.2)
        continue
    print('video channel %d ready.' % ch)


if __name__ == "__main__":
    for k,(p,t,z,b) in enumerate([(0,0,5, False), (20,20,20,True)]):
        print('%d-1: r p t z =' % (k), get_ptz())
        set_ptz(p, t, z, block=b)
        print('%d-2: r p t z =' % (k), get_ptz())
        time.sleep(2)
        print('%d-3: r p t z =' % (k), get_ptz())

    set_ptz(0, 0, 1)
    while True:
        for ch in range(video_channel_num):
            win_name = 'video channel-%d' % ch
            ret, image = get_image(video_channel=ch, flip=False)
            if ret and image is not None:
                module_result.show_image(win_name, image)
                print('channel=%d, image.shape =' % ch, image.shape)

        time.sleep(0.2)