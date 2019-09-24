import time
import utils
import threading
import cv2 as cv


lock = threading.Lock()


class_names = utils.get_config('class_names')
def create_object(class_name='bird',
                  state='static',
                  time_stamp=0.0,
                  channel=0,
                  bbox=[200,200,400,400],
                  score=1.0):
    """ bbox = [xmin,ymin,xmax,ymax] """
    if type(class_name) != str:
        class_id = int(class_name)
        if class_id >= len(class_names):
            class_id = len(class_names) - 1
        elif class_id < 0:
            class_id = 0
        class_name = class_names[int(class_id)]
    obj = {
            'class_name': class_name,
            'state': state,
            'time_stamp': time_stamp,
            'channel': channel,
            'bbox': bbox,
            'center': [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2],
            'score': score
           }
    return obj


result = {
            'panoramic': [create_object(), create_object()],
            'ptz': [create_object(), create_object()]
         }


def get_result():
    global result
    with lock:
        return result.copy()


def set_result(new_result):
    global result
    with lock:
        result = new_result.copy()


from queue import Queue
queue_image = Queue(maxsize=4)


def show_image(win_name, image, objs=[]):
    queue_image.put((win_name, image, objs))


def show_image_cv(win_name, image, objs):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        img = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # img = image.copy()
    else:
        img = image.copy()
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    line_width = utils.get_draw_line_width(img)
    colors = {'bird': (0,255,0), 'person': (0,0,255), 'unknown': (0,0,0)}
    for obj in objs:
        xmin, ymin, xmax, ymax = tuple(obj['bbox'])
        color = colors[obj['class_name']]
        cv.rectangle(img, (xmin,ymin), (xmax,ymax), color, line_width)
    cv.imshow(win_name, img)


def proc_show_image(*args, **kwargs):
    win_name, image, objs = queue_image.get()
    show_image_cv(win_name, image, objs)
    while True:
        try:
            win_name, image, objs = queue_image.get(block=False)
            show_image_cv(win_name, image, objs)
        except:
            key = cv.waitKey(10)
            if key == 32:
                while cv.waitKey() != 32:
                    continue
            
            
        


thread_show_image = utils.my_thread(proc_show_image)
thread_show_image.start()
        

if __name__ == "__main__":
    t0 = time.time()
    for k in range(int(1e5)):
        result = get_result()
        result['test_%03d' % (k%100)] = 'test_%03d' % (k)
        set_result(result)
    t1 = time.time()

    print('time elapsed:', t1 - t0)
    result = get_result()
    print(result)

    thread_show_image.join()