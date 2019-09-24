

#import numpy as np
import os
#import threading
import time
#import utils
#import module_result
import numpy as np
#from module_result import create_object
#import module_camera
import cv2 as cv
#import module_detection
x = 2

y = round(x)

print(type(x), x, y)
print(2**3)

def get_split_bbox(image, dst_rows_min=300, dst_cols_min=300, overlap_ratio_min = 0.10):
    src_rows = image.shape[0]
    src_cols = image.shape[1]
    bboxes = []
    
    for level in range(8):
        dst_rows = dst_rows_min * 2**level
        dst_cols = dst_cols_min * 2**level

        print(level, dst_rows, dst_cols)

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

if __name__ == "__main__":
    path='../input/test_pics/'   #要裁剪的图片所在的文件夹
    filename='w20190805084121403_1.jpg'    #要裁剪的图片名
    image = cv.imread(path+filename,1)
    bboxes = get_split_bbox(image, dst_rows_min=600, dst_cols_min=1024)
    print("len(bboxes)", len(bboxes))

    for i in range(len(bboxes)):
        print(bboxes[i])