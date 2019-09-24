#!/usr/bin/python3.5

import numpy as np
import os
import sys
import inspect
import tensorflow as tf
# import matplotlib
#
# matplotlib.use('Agg')

# from matplotlib import pyplot as plt
from PIL import Image
import threading
from queue import Queue
import time
import cv2 as cv
import utils

# import module_panoramic

# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util


# usr=============================================
score_thresh = utils.get_config('score_thresh')
# ================================================

input_image_rows = utils.get_config('model')['input_image_rows']
input_image_cols = utils.get_config('model')['input_image_cols']


def run_inference_for_image_batch(images_batch, sess, tensor_dict, image_tensor):
    # # Get handles to input and output tensors
    # ops = tf.get_default_graph().get_operations()
    # all_tensor_names = {output.name for op in ops for output in op.outputs}
    # tensor_dict = {}
    # for key in [
    #     'num_detections', 'detection_boxes', 'detection_scores',
    #     'detection_classes', 'detection_masks'
    # ]:
    #     tensor_name = key + ':0'
    #     if tensor_name in all_tensor_names:
    #         tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
    #             tensor_name)
    # if 'detection_masks' in tensor_dict:
    #     # The following processing is only for single image
    #     detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    #     detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    #     # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    #     real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    #     detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    #     detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    #         detection_masks, detection_boxes, image.shape[0], image.shape[1])
    #     detection_masks_reframed = tf.cast(
    #         tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    #     # Follow the convention by adding back the batch dimension
    #     tensor_dict['detection_masks'] = tf.expand_dims(
    #         detection_masks_reframed, 0)
    # image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: images_batch})
    # # all outputs are float32 numpy arrays, so convert types as appropriate
    # output_dict['num_detections'] = int(output_dict['num_detections'][0])
    # output_dict['detection_classes'] = output_dict[
    #     'detection_classes'][0].astype(np.uint8)
    # output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    # output_dict['detection_scores'] = output_dict['detection_scores'][0]
    # if 'detection_masks' in output_dict:
    #     output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


END_FLAG = 'end'


class NetThread(threading.Thread):
    def __init__(self, threadId, threadName, netModelPath):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.threadName = threadName
        self.netModelPath = netModelPath
        self.isRun = False
        self.inQueue = Queue()
        self.outQueue = Queue()
        self.lock = threading.Lock()

    def mprint(self, str):
        s = inspect.stack()
        module_name = inspect.getmodulename(s[1][1])  # sys._getframe().f_code.co_filename
        call_func_name = s[1][3]
        print('[{0}:thread-{1}-{2}:{3}] {4}'.format(
            module_name, self.threadId, self.threadName, call_func_name, str))

    def callNet(self, images_list):
        self.mprint('images_list type: {}'.format(type(images_list)))
        self.mprint('images_list len: {}'.format(len(images_list)))
        if images_list is None or len(images_list) == 0:
            return []
        with self.lock:
            self.inQueue.put(images_list)
            rst = self.outQueue.get()  # block
        self.mprint('done')
        return rst

    def begin(self):
        self.mprint(' ')
        self.lock.acquire()
        self.isRun = True
        self.start()
        self.lock.release()

        self.callNet([np.zeros((input_image_rows, input_image_cols, 3))])

    def end(self):
        self.mprint(' ')
        self.lock.acquire()
        self.isRun = False
        self.inQueue.put(END_FLAG)
        self.lock.release()

    # run SSD module
    def run(self):
        self.mprint('start--------')
        self.mprint('net model path: {}'.format(self.netModelPath))

        ckpt_path = os.path.join(self.netModelPath, 'frozen_inference_graph.pb')

        detection_graph = tf.Graph()
        sess = tf.Session(graph=detection_graph)

        with detection_graph.as_default():
            # initial
            self.mprint('initial graph')
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.mprint('load ckpt done')
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            with tf.Session() as sess:
                # call loop--------------------------
                self.mprint('graph initial done, enter call loop')
                while (self.isRun):
                    images_list = self.inQueue.get()  # block
                    if images_list == END_FLAG:
                        break

                    # prepare numpy.array images batch
                    num_imgs = len(images_list)
                    image_batch_np = np.zeros((num_imgs, input_image_rows, input_image_cols, 3))
                    t0 = time.time()
                    for k in range(num_imgs):
                        image_batch_np[k] = cv.resize(images_list[k],
                                                      (input_image_cols, input_image_rows),
                                                      interpolation=cv.INTER_AREA)[:, :, ::-1]
                    t1 = time.time()
                    self.mprint('resize images time cost {}s'.format(t1 - t0))

                    # Actual detection.
                    time_start = time.time()
                    output_dict = run_inference_for_image_batch(image_batch_np, sess, tensor_dict, image_tensor)
                    time_end = time.time()
                    self.mprint('inference image batch time cost: {}s'.format(time_end - time_start))

                    # boxes_batch[k] is [[xmin,ymin,xmax,ymax,class_id,probility], [...],...]
                    boxes_batch = []
                    for idx_batch in range(num_imgs):
                        boxes = []
                        im_rows = images_list[idx_batch].shape[0]
                        im_cols = images_list[idx_batch].shape[1]
                        for i in range(output_dict['detection_boxes'][idx_batch].shape[0]):
                            if output_dict['detection_scores'][idx_batch] is None \
                                    or output_dict['detection_scores'][idx_batch][i] > score_thresh:
                                box = output_dict['detection_boxes'][idx_batch][i].tolist()
                                (box[0], box[2], box[1], box[3]) = (int(box[1] * im_cols), int(box[3] * im_cols),
                                                                    int(box[0] * im_rows), int(box[2] * im_rows))
                                box.append(
                                    int(round(output_dict['detection_classes'][idx_batch][i])))  # box[4]: class id
                                box.append(output_dict['detection_scores'][idx_batch][i])  # box[5]: probility
                                boxes.append(box)
                        boxes_batch.append(boxes)
                    self.mprint('result: {}'.format(boxes_batch))
                    self.outQueue.put(boxes_batch)
                # call loop end--------------------------

        self.mprint('over------')

    ## run yolov3 model
    # def run(self, conf_thresh=0.25, iou_thresh=0.3):
    #     '''
    #
    #     :param namefile:
    #     :param inputsize: Imgsize to input to the model:(w,h)
    #     :param conf_thresh:
    #     :param iou_thresh:
    #     :return:
    #     '''
    #     self.mprint('start--------')
    #     self.mprint('net model path: {}'.format(self.netModelPath))
    #     ckpt_path = os.path.join(self.netModelPath,
    #                              'bird.pb')
    #     input_size = (input_image_cols, input_image_rows)
    #     detection_graph = tf.Graph()
    #     sess = tf.Session(graph=detection_graph)
    #
    #     with detection_graph.as_default():
    #         # initial
    #         self.mprint('initial graph')
    #         od_graph_def = tf.GraphDef()
    #         with tf.gfile.GFile(ckpt_path, 'rb') as fid:
    #             serialized_graph = fid.read()
    #             od_graph_def.ParseFromString(serialized_graph)
    #             tf.import_graph_def(od_graph_def, name='')
    #         self.mprint('load ckpt done')
    #         # Get handles to input and output tensors
    #         boxes = tf.get_default_graph().get_tensor_by_name("output_boxes:0")
    #         inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")
    #
    #         with tf.Session() as sess:
    #             # call loop--------------------------
    #             self.mprint('graph initial done, enter call loop')
    #             while (self.isRun):
    #                 images_list = self.inQueue.get()  # block
    #                 if images_list == END_FLAG:
    #                     break
    #
    #                 # prepare numpy.array images batch
    #                 num_imgs = len(images_list)
    #                 image_batch_np = np.zeros((num_imgs, input_image_rows, input_image_cols, 3))
    #                 t0 = time.time()
    #                 for k in range(num_imgs):
    #                     image_batch_np[k] = utils.resize_cv2(images_list[k], input_size)[:, :, ::-1]
    #                 t1 = time.time()
    #                 self.mprint('resize images time cost {}s'.format(t1 - t0))
    #
    #                 # Actual detection.
    #                 time_start = time.time()
    #                 # output_dict = run_inference_for_image_batch(image_batch_np, sess, boxes, inputs)
    #                 detected_boxes = sess.run(boxes, feed_dict={inputs: image_batch_np})
    #                 filtered_boxes = utils.non_max_suppression(detected_boxes,
    #                                                            confidence_threshold=conf_thresh,
    #                                                            iou_threshold=iou_thresh)
    #                 time_end = time.time()
    #                 self.mprint('inference image batch time cost: {}s'.format(time_end - time_start))
    #
    #                 # boxes_batch[k] is [[xmin,ymin,xmax,ymax,class_id,probility], [...],...]
    #                 boxes_batch = []
    #                 for i, batch in enumerate(filtered_boxes):
    #                     boxes_list = []
    #                     for cls, bboxs in batch.items():
    #                         for box, score in bboxs:
    #                             box = utils.convert_to_original_size(box, np.array(input_size),
    #                                                                  np.array(images_list[i].shape[:2][::-1]))
    #                             box.extend([cls, score])
    #                             boxes_list.append(box)
    #                     boxes_batch.append(boxes_list)
    #                 self.mprint('result: {}'.format(boxes_batch))
    #                 self.outQueue.put(boxes_batch)
    #             # call loop end--------------------------
    #
    #     self.mprint('over------')


def show_image_cv(win_name, image, objs):
    img = image.copy()
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    line_width = round((image.shape[0] + image.shape[1]) / 1500)
    line_width = max(line_width, 1)

    colors = [(0, 255, 0), (0, 0, 255), (0, 0, 0)]

    # using image path
    for obj in objs:
        xmin, ymin, xmax, ymax = tuple(obj[0:4])
        obj_index = min(obj[4], 2)
        if obj_index < 0: obj_index = 0
        if obj_index > 2: obj_index = 2
        obj_index = round(obj_index)
        color = colors[obj_index]
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_width)
        cv.putText(img, '{}:{:.2f}%'.format(obj[4], obj[5] * 100),
                   (xmin, ymin), cv.FONT_HERSHEY_PLAIN, 1, color, 1)
    cv.imshow(win_name, img)
    # cv.waitKey(0)


if __name__ == '__main__':
    modelDir = '/home/tk/Desktop/'

    net = NetThread(-1, 'thread_test_net', modelDir)
    net.begin()

    images_path = ['/home/tk/Desktop/bird/20190805/seg/JPEGImages/w20190719093725711_127_epoch0_aug0x0.jpg',
                   '/home/tk/Desktop/bird/20190917/JPEGImages/w20190705043739842_1_epoch0_aug0x1.jpg',
                   '/home/tk/Desktop/bird/20190805/seg/JPEGImages/w20190805083851060_120_epoch0_aug0x0.jpg',
                   '/home/tk/Desktop/bird/20190917/JPEGImages/w20190705043739842_3_epoch0_aug0x0.jpg']
    images_list = []
    for img_path in images_path:
        images_list.append(cv.imread(img_path))
    t0 = time.time()
    rst_images = net.callNet(images_list)
    t1 = time.time()
    # for i in range(10):
    #     rst_images = net.callNet(images_list)
    # t2 = time.time()
    # print("t1 - t0 = ", t1-t0)
    # print("t2 - t1 = ", t2-t1)
    # print("t2 - t0 = ", t2-t0)

    for k in range(len(images_list)):
        show_image_cv(images_path[k], images_list[k], rst_images[k])
    cv.waitKey()
    cv.destroyAllWindows()

    # path='../input/test_pics/'   #要裁剪的图片所在的文件夹
    # filename='w20190805084121403_1.jpg'    #要裁剪的图片名
    # image = cv.imread(path+filename,1)
    # t3 = time.time()
    # bboxes = module_panoramic.get_split_bbox(image, dst_rows_min=300, dst_cols_min=533)
    # print("len(bboxes)", len(bboxes))
    # objs = module_panoramic.detect_objs_in_rois(image, rois)
    # t4 = time.time()
    # print("t4-t3=", t4-t3)

    net.end()
