#-*-coding:utf_*_
import numpy as np
import cv2
import os

filename = "../input/test_pics/111.jpg"
rows_th=int(filename.split("/")[-1].split(".")[0])
print(rows_th)
"""
输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
"""
info = 1
def clip_one_picture(path,filename,cols,rows):
    img=cv2.imread(path+filename,1)##读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
    cv2.imwrite(path + str(info) + '.jpg', img)
    sum_rows=img.shape[0]   #高度
    sum_cols=img.shape[1]    #宽度
    save_path=path+"crop{0}_{1}/".format(cols,rows)  #保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols/cols),int(sum_rows/rows)))

    for i in range(int(sum_cols/cols)):
        for j in range(int(sum_rows/rows)):
            cv2.imwrite(save_path+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1],img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:])
            #print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
    print("裁剪完成，得到{0}张图片.".format(int(sum_cols/cols)*int(sum_rows/rows)))
    print("裁剪所得图片的存放地址为：{0}".format(save_path))


"""调用裁剪函数示例"""
path='../input/test_pics/'   #要裁剪的图片所在的文件夹
filename='w20190805084121403_1.jpg'    #要裁剪的图片名
cols=1024        #小图片的宽度（列数）
rows=600        #小图片的高度（行数）
clip_one_picture(path,filename,1024,600)





"""
输入：图片路径(path+filename)，裁剪所的图片的列的数量、行的数量
输出：无
"""

def merge_picture(merge_path,num_of_cols,num_of_rows):
    filename=file_name(merge_path,".jpg")
    shape=cv2.imread(filename[0],1).shape    #三通道的影像需把-1改成1,四通道为-1
    cols=shape[1]
    rows=shape[0]
    channels=shape[2]
    dst=np.zeros((rows*num_of_rows,cols*num_of_cols,channels),np.uint8)
    "w20190805084121403_1_0_1.jpg"
    for i in range(len(filename)):
        img=cv2.imread(filename[i],1)
        cols_th=int(filename[i].split("_")[-1].split('.')[0])
        rows_th=int(filename[i].split("_")[-2])
        roi=img[0:rows,0:cols,:]
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols,:]=roi
    cv2.imwrite(merge_path+"merge.jpg",dst)

"""遍历文件夹下某格式图片"""

def file_name(root_path,picturetype):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1]==picturetype:
                filename.append(os.path.join(root,file))
    return filename


"""调用合并图片的代码"""

merge_path="../input/test_pics/crop1024_600/"   #要合并的小图片所在的文件夹
num_of_cols=4    #列数
num_of_rows=3     #行数
merge_picture(merge_path,num_of_cols,num_of_rows)
