import numpy as np
from multiprocessing import Process
import os
from api import PRN
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from utils.render_app import get_depth_image
import ast
import argparse
import time


def run(imgList, args):
    # ---- init PRN
    prn = PRN(is_dlib = args.isDlib)

    # print("pid is {}, the length of imgs is {}.".format(os.getpid(), len(imgList)))
    for i, image_path in enumerate(imgList):    
        if i % 5 == 0:
            print('#{} => Finished {}/{}'.format(os.getpid(), i, len(imgList)))

        rela_path = '/'.join(image_path.split('/')[-3:])
        abs_path = os.path.join(args.dstRoot, rela_path)
        svd_path = abs_path.replace('.jpg', '_dep.jpg')

        if os.path.exists(svd_path):
            # print('{} exists'.format(svd_path))
            continue
        # else:
            # print('{} doesn\'t exists'.format(svd_path))
        
        dir_path = '/'.join(abs_path.split('/')[:-1])
        # box_path = image_path.replace('.jpg', '.dat')
        # if not os.path.exists(box_path):
            # print('{} dosen\'t exist.'.format(box_path))
            # continue

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


        # read image
        image = imread(image_path)
        
        if args.isResize:
            image = resize(image, (args.height, args.width)) * 255.
            image = image.astype(np.uint8)

        [h, w, c] = image.shape
        if c>3:
            image = image[:,:,:3]

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size> 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            pos = prn.process(image) # use dlib to detect face
        else:
            box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
            # with open(box_path, 'r') as f:
            #     box = f.readlines()
            #     x1, y1, ww, hh = box
            #     x1 = float(x1.strip('\n'))
            #     y1 = float(y1.strip('\n'))
            #     ww = float(ww.strip('\n'))
            #     hh = float(hh.strip('\n'))
            #     x1, y1, x2, y2 = int(x1), int(y1), int(x1 + ww), int(y1 + hh)
            #     box = np.array([x1, x2, y1, y2])
            pos = prn.process(image, box)
            
        # image = image/255.
        if pos is None:
            continue

        # 3D vertices
        vertices = prn.get_vertices(pos)
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        if args.isImage:
            imsave(abs_path, (image).astype(np.uint8))

        # vertices.shape:(43867, 3)
        depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        dep_img_path = svd_path
        imsave(dep_img_path, depth_image)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    imgList = []
    for root, dirs, files in os.walk(args.imgRoot):
        for file in files:
            if not file.endswith('scene.jpg') and file.endswith('.jpg') and file.split('_')[-3] == '1':
                imgList.append(os.path.join(root, file))
    
    print('==> There are {} files to be processed.'.format(len(imgList)))

    fractor = len(imgList) // args.workers
    for i in range(args.workers):
        if i != args.workers - 1:
            lst = imgList[fractor * i:fractor * (i + 1)]
            p = Process(target=run, args=(lst, args))
            p.start()
        else:
            lst = imgList[fractor * i:]
            p = Process(target=run, args=(lst, args))
            p.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 3D image using multiprocess.')
    parser.add_argument('--imgRoot', default='-1', type=str, required=True, 
                        help='process imgs with this dir and subdirs')
    parser.add_argument('--dstRoot', default='-1', type=str, required=True, 
                        help='save depth image following the same dir structures as imgRoot')
    parser.add_argument('--height', default=256, type=int,
                        help='the height of output depth image.')
    parser.add_argument('--width', default=256, type=int,
                        help='the width of output depth image.')
    parser.add_argument('--workers', default=8, type=int, required=True,
                        help='the num of processes.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--isImage', default=False, type=ast.literal_eval,
                        help='whether to save input image')
    parser.add_argument('--isResize', default=True, type=ast.literal_eval,
                        help='whether to resize input image')
    main(parser.parse_args())