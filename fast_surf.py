#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
from matplotlib import pyplot as plt



#bFirst = True
thre = 70
MIN_MATCH_COUNT = 10
#1.打开视频或摄像头并获取头像
cap = cv2.VideoCapture("/home/houxueyuan/detect/test_vedio/40mniao.dav")
vedio_fps = cap.get(cv2.CAP_PROP_FPS)  #CV_CAP_PROP_FPS = 5
print("fps", vedio_fps)
#2.逐帧或隔帧配准查分，获取运动目标
curr_frame = None
prev_frame = None
not_find = 0
frame_count = 0
#ret, frame = cap.read()
#此时的frame为视频的第一帧，因为此时prev_frame=None,仅做处理，不做查分
while(True):
	
	start_time = cv2.getTickCount()

	ret, frame = cap.read()	#后续操作，需要判定是否读视频结束，即判定ret是否未true

	time_read = cv2.getTickCount()
	time_read_cost = 1000 * (time_read - start_time)/cv2.getTickFrequency()
	print("time_read_cost = ", time_read_cost)

	if ret == False:
		print("Read frame failed!")
		break
	curr_frame = frame
	frame_count = frame_count + 1
	#2.1 获取灰度图像,FAST无需在灰度图像上执行
	curr_frame_1 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
	#2.2 提取FAST特征
	fast = cv2.FastFeatureDetector_create(thre)
	curr_kp = fast.detect(curr_frame_1, None)

	time_fast = cv2.getTickCount()
	time_fast_cost = 1000 * (time_fast - time_read)/cv2.getTickFrequency()
	print("time_fast_cost = ", time_fast_cost)

	print("curr_kp numbers:", len(curr_kp))
	#2.3 获得surf描述子
	surf = cv2.xfeatures2d.SURF_create()
	curr_kp, curr_des = surf.compute(curr_frame_1, curr_kp)

	time_surf = cv2.getTickCount()
	time_surf_cost = 1000 * (time_surf - time_fast)/cv2.getTickFrequency()
	print("time_fast_cost = ", time_surf_cost)

	#curr_frame is the first frame
	if curr_frame is not None and prev_frame is None:	
		prev_frame = curr_frame.copy()
		prev_frame_1 = curr_frame_1.copy()
		prev_des = curr_des.copy()
		prev_kp = curr_kp
		continue
			
	if curr_frame is not None and prev_frame is not None:	#curr_frame is not the first frame		
		#2.4 获得匹配特征点，并提取最优配对
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(prev_des,curr_des,k=2)

		time_match = cv2.getTickCount()
		time_match_cost = 1000 * (time_match - time_surf)/cv2.getTickFrequency()
		print("time_match_cost = ", time_match_cost)

		# store all the good matches as per Lowe's ratio test.
		#2.5 获取排在前N个的最优匹配特征点
		good = []
		for m,n in matches:
			if m.distance < 0.25*n.distance:
				good.append(m)		
		#2.6 获取图像1到图像2的投影映射矩阵 尺寸为3*3
		if len(good) > MIN_MATCH_COUNT:
			# 通过距离近的描述符 找到两幅图片的关键点
			src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
			dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			#matchesMask = mask.ravel().tolist()
			#print("prev_frame.shape", prev_frame.shape)
			rows,cols, _ = prev_frame.shape
			# 计算第二张图相对于第一张图的畸变
			#2.7 图像配准
			#pts = np.float32([[0, 0], [0, h], [w, 0]]).reshape(-1, 1, 2)
			#dst = cv2.perspectiveTransform(pts, M)
			#curr_frame = cv2.polylines(curr_frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
			dst = cv2.warpPerspective(prev_frame, M, (cols, rows))
			#print("dst.shape", dst.shape)
			#cv2.imshow("dst",dst)
			#c = cv2.waitKey(0)
		#else:
			#matchesMask = None
		time_11 = cv2.getTickCount()
		time_11_cost = 1000 * (time_11 - time_match)/cv2.getTickFrequency()
		print("time_11_cost = ", time_11_cost)

		
		#2.8 做帧差分  dst和curr_frame_1
		diff = cv2.absdiff(curr_frame, dst)
		#2.9 二值化
		ret, binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
		#2.10 检索连通域绘制矩形框
		m_BiLi = 0.999 	#由于两幅图配准，边缘不会一致，因此对原图大小0.8的比例中搜索检测到的目标
		curr_frame_temp = curr_frame.copy()  #用于显示结果
		banary_temp = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
		
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
		banary_temp_dilate = cv2.morphologyEx(banary_temp, cv2.MORPH_DILATE, kernel)
		banary_temp_dilate, contours, hierarchy = cv2.findContours(banary_temp_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		#contours, hierarchy = cv2.findContours(banary_temp_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		
		time_22 = cv2.getTickCount()
		time_22_cost = 1000 * (time_22 - time_11)/cv2.getTickFrequency()
		print("time_22_cost = ", time_22_cost)
		

		print("len(contours) = ", len(contours))
		if (len(contours) < 1) :
			not_find = not_find + 1

		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(curr_frame_temp,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.drawContours(curr_frame_temp, contours, -1, (0,0,255), 1)
		result = cv2.flip(curr_frame_temp, 0)  #垂直翻转
		#显示测试结果
		end_time = cv2.getTickCount()
		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.startWindowThread()
		cv2.imshow("result",result)

		prev_frame = curr_frame.copy()
		
		time_cost = 1000 * (end_time - start_time)/cv2.getTickFrequency()
		print("time_cost = ", time_cost)
		
		c = cv2.waitKey(1)
print("not_find = ", not_find)
print("frame_count = ", frame_count)

cap.release()
cv2.destroyAllWindows()


