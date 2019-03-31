from config import debugging


import cv2

import numpy as np

from ai import go_head, go_tail, go_baby, go_gird
import tricks as t
import os.path
sketch_path = 'test.jpg'
dir = os.path.dirname(sketch_path)
name = os.path.basename(sketch_path).split('.')[0]

baby_path = os.path.join(dir, '{}_baby.jpg'.format(name))
result_path = os.path.join(dir, '{}_result.jpg'.format(name))
composition_path = os.path.join(dir, '{}_composition.jpg'.format(name))
icon_path = os.path.join(dir, '{}_icon.jpg'.format(name))
local_hint_path = os.path.join(dir, '{}_local_hint.jpg'.format(name))
sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)

points = [
    [0.5, 0.25, int('77', 16), int('ee', 16), int('00', 16), 0],
    [0.5, 0.75, int('00', 16), int('11', 16), int('cc', 16), 0],
]
# lineColor = np.array(['f', 'F', '00'])

#####
for _ in range(len(points)):
    points[_][1] = 1 - points[_][1]

sketch_1024 = t.k_resize(sketch, 64)
sketch_256 = t.mini_norm(t.k_resize(t.min_k_down(sketch_1024, 2), 16))
sketch_128 = t.hard_norm(t.sk_resize(t.min_k_down(sketch_1024, 4), 32))
print('sketch prepared')

baby = go_baby(sketch_128, t.opreate_normal_hint(t.ini_hint(sketch_128), points, type=0, length=1))
baby = t.de_line(baby, sketch_128)
for _ in range(16):
    baby = t.blur_line(baby, sketch_128)
baby = go_tail(baby)
baby = t.clip_15(baby)
cv2.imwrite(baby_path, baby)
print('baby born')
# composition = go_gird(sketch=sketch_256, latent=t.d_resize(baby, sketch_256.shape), hint=t.ini_hint(sketch_256))
# # composition = t.emph_line(composition, t.d_resize(t.min_k_down(sketch_1024, 2), composition.shape), lineColor)
# composition = go_tail(composition)
# cv2.imwrite(composition_path, composition)
composition  = cv2.imread('composition.jpg',)

print('composition saved')
# print(t.opreate_normal_hint(t.ini_hint(sketch_1024), points, type=2, length=2))
result = go_head(
    sketch=sketch_1024,
    global_hint=t.k_resize(composition, 14),
    local_hint=t.ini_hint(sketch_1024),
    global_hint_x=t.k_resize(composition, 14),
    alpha=1
)
result = go_tail(result)
cv2.imwrite(result_path, result)
cv2.imwrite(icon_path, t.max_resize(result, 128))
