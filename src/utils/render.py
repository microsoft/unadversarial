try:
    import torch
    import mitsuba 
    import os
    if not 'GPU_MODE' in os.environ:
        raise ValueError('Must set GPU_MODE environment variable')
    if os.environ['GPU_MODE'] == '1':
        mitsuba.set_variant("gpu_rgb")
    elif os.environ['GPU_MODE'] == '0':
        mitsuba.set_variant("packet_rgb")
    else:
        raise ValueError('GPU_MODE must be 0 or 1')

    from mitsuba.core import Thread, Bitmap, LogLevel, Struct, ScalarTransform4f, set_thread_count
    from random import random
    from mitsuba.core.xml import load_string, load_dict
    from mitsuba.python.util import traverse
    from time import time
    import numpy as np
    from multiprocessing import Process
    set_thread_count(1)
    t = Thread.thread().logger()
    t.set_log_level(LogLevel.Error)
    t.clear_appenders()
except Exception as e:
    print("Mitsuba not installed, rendering not supported...")

def sphere_sample(scale):
    real_scale = np.random.uniform(*scale)
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    return (real_scale * vec).tolist()

def render(scene_dict, info, in_q, out_q, dones_sh, texture_sh, render_sh, uv_sh, loop_forever=True):
    a = time()
    scene = load_dict(scene_dict)
    print(f"Scene loading time: {time() - a:.2f}s")
    bsdf_params = traverse(scene.shapes()[0].bsdf())
    integrator = scene.integrator()

    # Numpy versions of the shared tensors
    texture_sh_np = texture_sh.numpy()
    render_sh_np = render_sh.numpy()
    uv_sh_np = uv_sh.numpy()
    while True:
        camera_pos = sphere_sample(info['scale_range'])
        camera = load_dict({
            "type": "perspective",
            "to_world" : ScalarTransform4f.look_at(origin=camera_pos, target=[0, 0, 0], up=[0, 0, 1]),
            "myfilm": {
                "type": "hdrfilm",
                "width": info['image_size'],
                "height": info['image_size']
            },
            "mysampler": {
                "type": "independent",
                "sample_count": info['samples']
            }})
        ind = in_q.get()
        tex = texture_sh_np[ind]
        bsdf_params['diffuse_reflectance.data'] = tex
        integrator.render(scene, camera)
        film = camera.film()
        im = film.bitmap(raw=False).split()
        im_arr = np.array(im[0][1].convert(Bitmap.PixelFormat.RGBA, Struct.Type.Float32, srgb_gamma=False))
        u = np.array(im[1][1].convert(Bitmap.PixelFormat.Y, Struct.Type.Float32, srgb_gamma=False))
        v = np.array(im[2][1].convert(Bitmap.PixelFormat.Y, Struct.Type.Float32, srgb_gamma=False))
        uv_arr = np.concatenate([u, v, np.zeros_like(u), (u + v > 0).astype(np.float32)], axis=2)
        render_sh_np[ind,...] = im_arr 
        uv_sh_np[ind,...] = uv_arr
        dones_sh[ind] = True
        if not loop_forever: 
            print("Done")
            return
