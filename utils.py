import argparse
import numpy as np
from PIL import Image
import time
from contextlib import contextmanager
import inspect
import threading
import cv2


def get_now_time_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_now_time_filename():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

@contextmanager
def time_elapsed(title):
    t0 = time.time()
    print('\n' + '-'*80)
    print(get_now_time_string(), title)
    yield
    print("{}: done in {:.3f}s".format(title, time.time() - t0))
    print('-'*80 + '\n')



def save_submission(df_sub, prefix='sub_'):
    filename = '../output/' + prefix + ".csv"
    df_sub.to_csv(filename, index=False)
    print('saved submission file:', filename)



import json
def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    config_dict = dict()
    
    try:
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
    except IOError:
        print("Error: load %s failed" % json_file)
    return config_dict


def get_config(key, filename='config.json'):
    try:
        val = get_config_from_json(filename)[key]
    except:
        val = None
        print('get config %s fail' % (key))
    
    return val



def get_function_name():
    return inspect.stack()[1][3]


class my_thread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def run(self):
        self.func(*self.args, **self.kwargs)


def get_draw_line_width(image):
    line_width = round(min(image.shape[0],image.shape[1])/200)
    line_width = max(line_width, 1)
    return line_width

def _iou(box1, box2)->float:
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(0,(int_x1 - int_x0)) * max(0,(int_y1 - int_y0))

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4)-> dict:
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    results = []
    for i, image_pred in enumerate(predictions):
        result = {}
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-
        1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]    # 得到该类的所有boxes
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]   # np.argsort() 对数列从小到大排序，返回序号
            cls_scores = cls_boxes[:, -1]  # 最后一列表示得分（置信度）
            cls_boxes = cls_boxes[:, :-1]  # 前四列是box位置信息

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]   # 先选择置信度最大的box和score作为基准
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]
        results.append(result)
    # print (results)
    return results

def load_names(file_name)->dict:
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def draw_boxes(boxes, img, cls_names, detection_size, is_letter_box_image):
    draw = ImageDraw.Draw(img)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    for cls, bboxs in boxes.items():
        color = colors[cls%6]
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.size),
                                           is_letter_box_image)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(
                cls_names[cls], score * 100), fill=color)

def draw_boxes_cv2(boxes:dict, img:np.ndarray, cls_names:dict, detection_size:tuple, ispadding=False):
    # draw = ImageDraw.Draw(img)
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    for cls, bboxs in boxes.items():
        color = colors[cls%6]
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size),
                                           np.array(img.shape[:2][::-1]),  # (h,w)->(w,h)
                                           ispadding)
            box = [max(1, box[0]), max(1, box[1]),
                   min(img.shape[1] - 1, box[2]), min(img.shape[0] - 1, box[3])]
            left_top, right_bottom = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(img,left_top, right_bottom,color,2)
            cv2.putText(img, '{}{:.2f}%'.format(cls_names[cls].strip(), score * 100),
                        left_top,cv2.FONT_HERSHEY_PLAIN , 1, color, 1)
            print('name:{0},\t location:{1[0]:>4d},{1[1]:>4d},{1[2]:>4d},{1[3]:>4d},\t confidence:{2:.2%}'
                  .format(cls_names[cls].strip(),box,score))

def convert_to_original_size(box: np.ndarray, size: np.ndarray, original_size: np.ndarray, is_letter_box_image=False)->list:
    if is_letter_box_image:
        box = box.reshape(2, 2)
        box[0, :] = letter_box_pos_to_original_pos(box[0, :], size, original_size)
        box[1, :] = letter_box_pos_to_original_pos(box[1, :], size, original_size)
    else:
        ratio = original_size / size
        box = box.reshape(2, 2) * ratio
    return [int(i) for i in box.reshape(-1)]

def letter_box_image(image: Image.Image, output_height: int, output_width: int, fill_value)-> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height)/image.size[1]
    width_ratio = float(output_width)/image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top:pad_top+fit_height, pad_left:pad_left+fit_width] = fit_image
    return to_return

def resize_cv2(image:np.ndarray, output_size:tuple, fill_value=128, ispadding=False)-> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """
    output_width, output_height = output_size[0], output_size[1]
    if ispadding:
        height_ratio = float(output_height)/image.shape[0]
        width_ratio = float(output_width)/image.shape[1]
        fit_ratio = min(width_ratio, height_ratio)
        fit_height = int(image.shape[0] * fit_ratio)
        fit_width = int(image.shape[1] * fit_ratio)
        fit_image = cv2.resize(image, output_size,cv2.INTER_LINEAR)

        if isinstance(fill_value, int):
            fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

        to_return = np.tile(fill_value, (output_height, output_width, 1))
        pad_top = int(0.5 * (output_height - fit_height))
        pad_left = int(0.5 * (output_width - fit_width))
        to_return[pad_top:pad_top+fit_height, pad_left:pad_left+fit_width] = fit_image
        return to_return
    else:
        return cv2.resize(image, output_size, cv2.INTER_LINEAR)

def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos


if __name__ == "__main__":
    with time_elapsed('test utls'):
        conf = get_config_from_json('config.json')
        print('config.json:\n', conf)
        
        conf = get_config_from_json('config1.json')
        print('config.json:\n', conf)

        val = get_config('batch_size')
        print('batch_size =', val)

        print('valid_ratio_min', get_config('valid_ratio_min'))

        val = get_config('abcdef')
        print('abcdef =', val)



