import socket
import threading
import struct
import time
import cv2
import numpy
import os
#tmp_buf = [1024*1024]
SEND_BUF_SIZE = 1024*1024*2 # 发送缓冲区的大小
RECV_BUF_SIZE = 1024*1024*2 # 接收缓冲区的大小

merge_flag = False
lock = threading.Lock()
frames = [None]*3

class Carame_Accept_Object:
    def __init__(self,S_addr_port=("",8880)):
        self.resolution=(640,480)       
        self.img_fps=15                 
        self.addr_port=S_addr_port
        self.Set_Socket(self.addr_port)
 
    
    def Set_Socket(self,S_addr_port):
        self.server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        bufsize = self.server.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print("Buffer size [Before]: %d" %bufsize)
    
        self.server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)
        
        bufsize = self.server.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print( "Buffer size [After]: %d" %bufsize)

        self.server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.server.bind(S_addr_port)
        self.server.listen(5)
        #print("the process work in the port:%d" % S_addr_port[1])
 
 
def check_option(object,client):
    
    info=struct.unpack('lhh',client.recv(8))
    if info[0]>888:
        object.img_fps=int(info[0])-888       
        object.resolution=list(object.resolution)
        
        object.resolution[0]=info[1]
        object.resolution[1]=info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        return 0



def merge_picture(merge_path,num_of_cols,num_of_rows):
    filename=file_name(merge_path,".jpg")
    shape=cv2.imread(filename[0],1).shape    #三通道的影像需把-1改成1,四通道为-1
    cols=shape[1]
    rows=shape[0]
    channels=shape[2]
    ret = False
    #print("cols, rows, channels", cols, rows, channels)
    #print("filename[0]", filename[0].split("/")[-1].split(".")[0])
    dst=numpy.zeros((rows*num_of_rows,cols*num_of_cols,channels),numpy.uint8)
    for i in range(len(filename)):
        #print("filename[]", os.path.splitext(filename[i])[0])
        img=cv2.imread(filename[i],1)
        file_num = int(filename[i].split("/")[-1].split(".")[0])
        #print("file_num = ", file_num)
        cols_th = int((file_num - 1) / 3)
        rows_th = int((file_num - 1) % 3)
        #print("cols_th:%d rows_th:%d" % (cols_th, rows_th))
        roi=img[0:rows,0:cols,:]
        dst[rows_th*rows:(rows_th+1)*rows,cols_th*cols:(cols_th+1)*cols,:]=roi
        with lock:
            frames[0] = dst
    #frame = frames[0]
    """
    cv2.namedWindow("frame[0]", cv2.WINDOW_NORMAL)
    cv2.imshow("frame[0]", frame)
    cv2.imwrite(merge_path+"merge.jpg",dst)
    merge_filename = merge_path + "merge.jpg"
    merge_image = cv2.imread(merge_filename, 1)
    cv2.namedWindow("merge.jpg", cv2.WINDOW_NORMAL)
    cv2.imshow("merge.jpg", merge_image)
    key = cv2.waitKey(1)
    key = cv2.waitKey(1)
    if key == 27:
        while cv2.waitKey() != 27:
            continue
    """
    

"""遍历文件夹下某格式图片"""

def file_name(root_path,picturetype):
    filename=[]
    for root,dirs,files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1]==picturetype:
                filename.append(os.path.join(root,file))
    return filename

def RT_Image(object,client,D_addr):
    cnt = 0
    save_path='../input/test_pics11/'
    while(1):
        cnt = cnt + 1
        #time.sleep(1)
        msg = client.recv(16)
        #print("msg length = ", len(msg))
        if len(msg) == 16:
            info = struct.unpack(">lld", msg)
            tmp_buf = b''
            tmp_buf1 = b''
            if(info[0]>12 or info[0]<1):
                #print("1111111111111111111111111111111111111111")
                continue
            
            #print("serial %d " % (info[0]))
                       
            buf_size = info[1]
            print("cnt:%d, continue serial:%d buf_size %d" % (cnt, info[0],buf_size))
            #print('timestamp =',time.time()-info[2])
            
            tmp_buf = client.recv(buf_size)
            while len(tmp_buf) < buf_size:
               #print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
               tmp_buf1 = client.recv(buf_size-len(tmp_buf))
               tmp_buf += tmp_buf1
               if len(tmp_buf) >= buf_size:
                   break
            #t0 = time.time()
            #print("len(tmp_buf)", len(tmp_buf))
            #print('timestamp1111 =',t0-info[2])
            #tmp_buf = tmp_buf[0:]
            image = numpy.fromstring(tmp_buf, numpy.uint8)
            decimg=cv2.imdecode(image,cv2.IMREAD_COLOR)

            cv2.imwrite(save_path + str(info[0]) + '.jpg', decimg)

            if info[0] == 12:
                with lock:
                    merge_flag = True
            else:
                with lock:
                    merge_flag = False
            print("merge_flag", merge_flag)
                #num_of_cols=4    #列数
                #num_of_rows=3     #行数
                #merge_picture(save_path,num_of_cols,num_of_rows)
                                
            #t1 = time.time()
            #print("imdecode time:", t1-t0)
            #decimg=cv2.imdecode(image,1)
            #cv2.imshow('serial-%d' % (info[0]), decimg)
            #if key == 27:
                #break
                
            #else:
                #continue

def get_image_from_socket(channel_num, flip = True):
    with lock:
        if frames[channel_num] is not None:
            ret = True
            frame = frames[channel_num]   
        else:
            ret = False
            frame = None
    return ret, frame

camera=Carame_Accept_Object()
client,D_addr=camera.server.accept()
clientThread=threading.Thread(None,target=RT_Image,args=(camera,client,D_addr,))
clientThread.start()


if __name__ == '__main__':
    #get_image_from_socket()
    """
    while True:
        ret, frame = get_image_from_socket()
        if ret:
            cv2.namedWindow("merge.jpg", cv2.WINDOW_NORMAL)
            cv2.imshow("merge.jpg", frame)
            key = cv2.waitKey(1)
            key = cv2.waitKey(1)
            if key == 27:
                while cv2.waitKey() != 27:
                    continue 
    """
    

    
    