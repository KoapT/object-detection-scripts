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
netdevlib = so('/home/houxueyuan/anti-bird/anti-bird/driver/bin/libnetdev.so')
print('netdevlib:', netdevlib)