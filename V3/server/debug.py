from config import *

import re
import os
import cv2
import time
import json


import datetime

import numpy as np


from tricks import *

from baby import go_baby
from tail import go_tail


sample_points = [
    [0.5, 0.25, '77', 'ee', '00', 0],
    [0.5, 0.75, '00', '11', 'cc', 0],
]


def a2(path, points, ):
        ID = path.split('/')[-1]
        sketch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        sketch_1024 = k_resize(sketch, 64)        
        sketch_256 = mini_norm(k_resize(min_k_down(sketch_1024, 2), 16))
        sketch_128 = hard_norm(sk_resize(min_k_down(sketch_1024, 4), 32))
        
        print('sketch prepared')
        
        cv2.imwrite('./sketch.128.jpg', sketch_128)
        cv2.imwrite( './sketch.256.jpg', sketch_256)
        baby = go_baby(sketch_128, opreate_normal_hint(ini_hint(sketch_128), points, type=0, length=1))
        baby = de_line(baby, sketch_128)
        for _ in range(16):
            baby = blur_line(baby, sketch_128)
        baby = go_tail(baby)
        baby = clip_15(baby)
        cv2.imwrite( './baby.' + ID + '.jpg', baby)


a2('./test.jpg', sample_points)


