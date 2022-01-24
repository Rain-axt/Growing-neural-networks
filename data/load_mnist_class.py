# -*-coding:utf8-*-

import os
import numpy as np
import argparse
import random
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser('load')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--eval_samples', type=int, default=400)
    parser.add_argument('--eval_files', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--tune_eval_dir', type=str)
    parser.add_argument('--tune_test_dir', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.tune_eval_dir):
        os.makedirs(args.tune_eval_dir)
    if not os.path.exists(args.tune_test_dir):
        os.makedirs(args.tune_test_dir)

    image_file = os.path.join(args.data_root, 'train-images-idx3-ubyte')
    label_file = os.path.join(args.data_root, 'train-labels-idx1-ubyte')

    fr = open(image_file)
    loaded = np.fromfile(file=fr, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
    fr.close()

    frl = open(label_file)
    labels = np.fromfile(file=frl, dtype=np.uint8)
    trY = labels[8:].reshape([60000]).astype(np.long)
    frl.close()

    data_order = list(range(60000))
    random.shuffle(data_order)

    # make train dataset
    train_dir = os.path.join(args.out_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    class_files = []
    class_number = {}
    class_ids = {}
    for i in range(args.num_class):
        out_file = os.path.join(train_dir, str(i) + '.pkl')
        fw = open(out_file, 'wb')
        class_files.append(fw)
        class_number[i] = 0
        class_ids[i] = set()
    train_count = 0
    for idx in data_order:
        img = (np.transpose(trX[idx, :, :, :], [2, 0, 1])) / 255 - 0.5
        lab = trY[idx]
        if class_number[int(lab)] < args.train_samples:
            pickle.dump([img, lab], class_files[int(lab)])
            class_number[int(lab)] += 1
            class_ids[int(lab)].add(idx)
            train_count += 1
        if train_count == (args.train_samples * args.num_class):
            break
    for i in range(args.num_class):
        class_files[i].close()

    # make eval dataset
    eval_dir = os.path.join(args.out_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    for i in range(args.eval_files):
        out_file = os.path.join(eval_dir, str(i) + '.pkl')
        fw = open(out_file, 'wb')
        eval_count = 0
        class_count = {}
        for j in range(args.num_class):
            class_count[j] = 0
        for idx in data_order:
            img = (np.transpose(trX[idx, :, :, :], [2, 0, 1])) / 255 - 0.5
            lab = trY[idx]
            if class_count[int(lab)] < args.eval_samples and idx not in class_ids[int(lab)]:
                pickle.dump([img, lab], fw)
                class_count[int(lab)] += 1
                class_ids[int(lab)].add(idx)
                eval_count += 1
            if eval_count == (args.num_class * args.eval_samples):
                break
        fw.close()

    # load tune eval and test set
    test_image_file = os.path.join(args.data_root, 't10k-images-idx3-ubyte')
    test_label_file = os.path.join(args.data_root, 't10k-labels-idx1-ubyte')

    fr = open(test_image_file)
    loaded = np.fromfile(file=fr, dtype=np.uint8)
    trX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)
    fr.close()

    frl = open(test_label_file)
    labels = np.fromfile(file=frl, dtype=np.uint8)
    trY = labels[8:].reshape([10000]).astype(np.long)
    frl.close()

    data_order = list(range(10000))
    random.shuffle(data_order)

    # make tune eval dataset
    class_files = []
    class_number = {}
    class_ids = {}
    for i in range(args.num_class):
        out_file = os.path.join(args.tune_eval_dir, str(i) + '.pkl')
        fw = open(out_file, 'wb')
        class_files.append(fw)
        class_number[i] = 0
        class_ids[i] = set()
    eval_count = 0
    for idx in data_order:
        img = (np.transpose(trX[idx, :, :, :], [2, 0, 1])) / 255 - 0.5
        lab = trY[idx]
        if class_number[int(lab)] < 500:
            pickle.dump([img, lab], class_files[int(lab)])
            class_number[int(lab)] += 1
            class_ids[int(lab)].add(idx)
            eval_count += 1
        if eval_count == (500 * args.num_class):
            break
    for i in range(args.num_class):
        class_files[i].close()
    # make tune test dataset
    class_files = []
    for i in range(args.num_class):
        out_file = os.path.join(args.tune_test_dir, str(i) + '.pkl')
        fw = open(out_file, 'wb')
        class_files.append(fw)
        class_number[i] = 0
    test_count = 0
    for idx in data_order:
        img = (np.transpose(trX[idx, :, :, :], [2, 0, 1])) / 255 - 0.5
        lab = trY[idx]
        if class_number[int(lab)] < 500 and idx not in class_ids[int(lab)]:
            pickle.dump([img, lab], class_files[int(lab)])
            class_number[int(lab)] += 1
            class_ids[int(lab)].add(idx)
            test_count += 1
        if test_count == (500 * args.num_class):
            break
    for i in range(args.num_class):
        class_files[i].close()
