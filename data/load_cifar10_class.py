# -*-coding:utf8-*-

import numpy as np
import argparse
import os
import pickle
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser('load')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--flatten', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=10)
    args = parser.parse_args()
    # build train file objects
    cls2file = {}
    train_dir = os.path.join(args.out_path, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    for i in range(args.num_class):
        out_class_file = os.path.join(train_dir, str(i) + '.pkl')
        fw = open(out_class_file, 'wb')
        cls2file[i] = fw
    # load data from file
    for fi in os.listdir(args.data_path):
        if 'data_batch_' not in fi:
            continue
        data_file = os.path.join(args.data_path, fi)
        with open(data_file, 'rb') as fr:
            sp = pickle.load(fr, encoding='bytes')
        a = sp[b'data']
        b = sp[b'labels']
        id_list = list(range(args.samples))
        random.shuffle(id_list)
        for i in range(args.samples):
            sp_id = id_list[i]
            if args.flatten == 1:
                img = a[sp_id].astype(np.float32).reshape([1, -1]) / 255 - 0.5
            else:
                img = a[sp_id].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[sp_id]).astype(np.long)
            cls = int(lab)
            pickle.dump([img, lab], cls2file[cls])
    # close files:
    for i in range(args.num_class):
        cls2file[i].close()
    # build dev-set, tiny-dev and test-set
    eval_dir = os.path.join(args.out_path, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    tiny_dir = os.path.join(args.out_path, 'tiny')
    if not os.path.exists(tiny_dir):
        os.makedirs(tiny_dir)
    test_dir = os.path.join(args.out_path, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    dev_file = os.path.join(args.data_path, 'test_batch')
    with open(dev_file, 'rb') as fr:
        sp = pickle.load(fr, encoding='bytes')
    a = sp[b'data']
    b = sp[b'labels']
    load_order = list(range(args.samples))
    random.shuffle(load_order)
    # num_samples = [500, 50, 450]
    num_samples = [600, 200, 200]
    loaded_set = set()
    for idp, out_dir in enumerate([eval_dir, tiny_dir, test_dir]):
        out_cls2file = {}
        class_count = {}
        # make dump files
        for i in range(args.num_class):
            out_class_file = os.path.join(out_dir, str(i) + '.pkl')
            fw = open(out_class_file, 'wb')
            out_cls2file[i] = fw
            class_count[i] = 0
        # load data
        for i in load_order:
            if i in loaded_set:
                continue
            if args.flatten == 1:
                img = a[i].astype(np.float32).reshape([1, -1]) / 255 - 0.5
            else:
                img = a[i].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[i]).astype(np.long)
            cls = int(lab)
            if class_count[cls] < num_samples[idp]:
                pickle.dump([img, lab], out_cls2file[cls])
                loaded_set.add(i)
                class_count[cls] += 1
        # close files
        for i in range(args.num_class):
            out_cls2file[i].close()
