import os
import json
from joblib import Parallel, delayed, parallel_backend
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse


def load_semantic_anno(semantic_txt):
    semantic_color = []
    obj_name_list = []
    color_2_name = {}
    color_2_id = {}
    with open(semantic_txt) as f:
        lines = f.readlines()[1:]
        for line in lines:
            obj_id = int(line.split(',')[0])
            color_str = line.split(',')[1]
            if len(color_str) != 6:
                color_str = '0' * (6 - len(color_str)) + color_str
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            obj_name = line.split(',')[2][1:-1]
            obj_name_list.append(obj_name)
            rgb_value = np.array([r, g, b], dtype=np.uint8).reshape(1, 3)
            semantic_color.append(rgb_value)
            color_2_name[(r, g, b)] = obj_name
            color_2_id[(r, g, b)] = obj_id
    return np.concatenate(semantic_color, axis=0), obj_name_list, color_2_name, color_2_id


def scene_proc(scene_input):
    scene_name = scene_input.split('/')[-1]
    scene_uid = scene_name.split('-')[1]
    sem_dir = scene_input + '/' + scene_uid + '.semantic'
    print(scene_name)

    # load obj semantics anno
    semantic_anno_color, obj_name_list, color_2_name, color_2_id = load_semantic_anno(sem_dir+'.txt')

    tgt_id2obj_id = {}
    # obj assignment and export
    semantic_anno_set = set(list(zip(*(semantic_anno_color.T))))
    for _i, sem in enumerate(tqdm(semantic_anno_set)):
        obj_name = color_2_name[(sem[0], sem[1], sem[2])]
        obj_id = color_2_id[(sem[0], sem[1], sem[2])]
        tgt_id2obj_id[_i+1] = (obj_id, obj_name)
    json.dump(tgt_id2obj_id, open(os.path.join(scene_input, 'tgt_id2obj_id.json'), 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./hm3d-train-annots', help='data root for hm-semantics data')
    args = parser.parse_args()
    scene_list = glob(args.data_root + '/*')
    with parallel_backend('multiprocessing', n_jobs=1):
        Parallel()(delayed(scene_proc)(scene) for scene in scene_list)