import time
import json
import threading
import module_panoramic
import module_ptz
import module_result
import utils

thread_panoramic = utils.my_thread(module_panoramic.proc_panoramic, 'proc_panoramic')
thread_panoramic.start()

thread_ptz = utils.my_thread(module_ptz.proc_ptz, 'proc_ptz')
thread_ptz.start()

thread_panoramic.join()
thread_ptz.join()
