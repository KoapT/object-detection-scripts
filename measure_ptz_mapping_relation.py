
import os
import time
import utils
import module_result
import module_camera
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 

def calc_shift(img0, img1):

    h, w, c = img1.shape
    min_match_count = 10

    surf = cv.xfeatures2d.SURF_create(1000)
    mask = np.zeros(img0.shape[0:2], dtype=np.uint8) + 255
    mask[0:h//5,w//2:] = 0
    mask[(4*h)//5:,0:w//2] = 0
    module_result.show_image('mask', mask)

    kp0, des0 = surf.detectAndCompute(img0, mask)
    kp1, des1 = surf.detectAndCompute(img1, mask)

    matcher = cv.BFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(des0, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    img_match = np.zeros((img0.shape[0], 2*img0.shape[1], img0.shape[2]), dtype=np.uint8)
    cv.drawMatches(img0, kp0, img1, kp1, good, img_match)
    module_result.show_image('img_match', img_match)
    if len(good) > min_match_count:
        src_pts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()

        pts = np.float32([[w//2, h//2]]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)
        shift = dst - pts
        shift = shift.reshape(-1)
        shift = (shift[0], shift[1])
        return shift
    
    return (0,0)


if __name__ == "__main__":
    list_zoom = []
    list_delta_xpixel = []
    list_delta_ypixel = []
    list_delta_pan   = []
    list_delta_tilt  = []

    for zoom in range(1,129,1):
        print('-'*80)
        print('zoom =', zoom)

        ptz_p0 = 10
        ptz_t0 = 8.5
        ptz_dp = 0.5
        ptz_dt = 0.5
        module_camera.set_ptz(ptz_p0, ptz_t0, zoom)
        time.sleep(1)
        r0, p0, t0, z0 = module_camera.get_ptz()
        print('p0, t0, z0 =', p0, t0, z0)
        r0, img0 = module_camera.get_image(flip=False)

        module_camera.set_ptz(ptz_p0+ptz_dp, ptz_t0+ptz_dt, zoom)
        time.sleep(1)
        r1, p1, t1, z1 = module_camera.get_ptz()
        print('p1, t1, z1 =', p1, t1, z1)
        r1, img1 = module_camera.get_image(flip=False)

        try:
            shift_x, shift_y = calc_shift(img0, img1)
        except:
            print('calc shift failed!!!')
            shift_x, shift_y = 0, 0
        print('shift x,y = ', shift_x, shift_y)

        list_zoom.append((z0+z1)/2)

        list_delta_xpixel.append( max(abs(shift_x), 1) )
        list_delta_ypixel.append( max(abs(shift_y), 1) )

        list_delta_pan .append(p1-p0)
        list_delta_tilt.append(t1-t0)

    print('\nlist_zoom:\n', list_zoom)
    print('\nlist_delta_xpixel:\n', list_delta_xpixel)
    print('\nlist_delta_ypixel:\n', list_delta_ypixel)
    print('\nlist_delta_pan:\n',    list_delta_pan)
    print('\nlist_delta_tilt:\n',   list_delta_tilt)

    plt.title("zoom vs pixels")
    plt.xlabel("zoom")
    plt.ylabel("pixels")
    plt.plot(list_zoom, list_delta_xpixel, label='xpixel')
    plt.plot(list_zoom, list_delta_ypixel, label='ypixel')
    plt.grid()
    plt.legend()
    plt.savefig("zoom_vs_pixel.png")

    plt.cla()
    plt.title("zoom vs angle")
    plt.xlabel("zoom")
    plt.ylabel("angle")
    plt.plot(list_zoom, list_delta_pan,  label='pan')
    plt.plot(list_zoom, list_delta_tilt, label='tilt')
    plt.grid()
    plt.legend()
    plt.savefig("zoom_vs_angle.png")

    arr_zoom = np.array(list_zoom)
    arr_x = np.array(list_delta_pan)  / np.array(list_delta_xpixel)
    arr_y = np.array(list_delta_tilt) / np.array(list_delta_ypixel)
    plt.cla()
    plt.title("zoom vs ratio=angle/pixels")
    plt.xlabel("zoom")
    plt.ylabel("ratio")
    plt.plot(arr_zoom, arr_x)
    plt.plot(arr_zoom, arr_y)
    plt.grid()
    plt.legend()
    plt.savefig("zoom_vs_ratio.png")

    print('arr_x:', arr_x)
    print('arr_y:', arr_y)




