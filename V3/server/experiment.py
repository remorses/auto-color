from ai import (
    go_baby,
    go_tail,
    go_head,
)




        improved_sketch = sketch.copy()
        improved_sketch = min_resize(improved_sketch, 512)
        improved_sketch = cv_denoise(improved_sketch)
        improved_sketch = sensitive(improved_sketch, s=5.0)
        improved_sketch = go_tail(improved_sketch)
        color_sketch = improved_sketch.copy()
        cv2.imwrite(room_path + '/sketch.improved.jpg', improved_sketch)
        cv2.imwrite(room_path + '/sketch.colorization.jpg', min_black(color_sketch))
        cv2.imwrite(room_path + '/sketch.rendering.jpg', eye_black(color_sketch))
        print('sketch improved')
        
        
        sketch_1024 = k_resize(sketch, 64)        
        sketch_256 = mini_norm(k_resize(min_k_down(sketch_1024, 2), 16))
        sketch_128 = hard_norm(sk_resize(min_k_down(sketch_1024, 4), 32))
        
        print('sketch prepared')
        if debugging:
            cv2.imwrite(room_path + '/sketch.128.jpg', sketch_128)
            cv2.imwrite(room_path + '/sketch.256.jpg', sketch_256)
        baby = go_baby(sketch_128, opreate_normal_hint(ini_hint(sketch_128), points, type=0, length=1))
        baby = de_line(baby, sketch_128)
        for _ in range(16):
            baby = blur_line(baby, sketch_128)
        baby = go_tail(baby)
        baby = clip_15(baby)
        if debugging:
            cv2.imwrite(room_path + '/baby.' + ID + '.jpg', baby)
        print('baby born')
        composition = go_gird(sketch=sketch_256, latent=d_resize(baby, sketch_256.shape), hint=ini_hint(sketch_256))
        if line:
            composition = emph_line(composition, d_resize(min_k_down(sketch_1024, 2), composition.shape), lineColor)
        composition = go_tail(composition)
        cv2.imwrite(room_path + '/composition.' + ID + '.jpg', composition)
        print('composition saved')
        print('method: ' + method)
        result = go_head(
            sketch=sketch_1024,
            global_hint=k_resize(composition, 14),
            local_hint=opreate_normal_hint(ini_hint(sketch_1024), points, type=2, length=2),
            global_hint_x=k_resize(reference, 14) if reference is not None else k_resize(composition, 14),
            alpha=(1 - alpha) if reference is not None else 1
        )
        result = go_tail(result)
        cv2.imwrite(room_path + '/result.' + ID + '.jpg', result)
        cv2.imwrite('results/' + room + '.' + ID + '.jpg', result)
        if debugging:
            cv2.imwrite(room_path + '/icon.' + ID + '.jpg', max_resize(result, 128))
