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
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--eval_samples', type=int, default=400)
    parser.add_argument('--eval_files', type=int, default=5)
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
            img = a[sp_id].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[sp_id]).astype(np.long)
            cls = int(lab)
            pickle.dump([img, lab], cls2file[cls])
    # close files:
    for i in range(args.num_class):
        cls2file[i].close()

    train_files = []
    test_file = ''
    for fi in os.listdir(args.data_path):
        if 'test_batch' in fi:
            data_file = os.path.join(args.data_path, fi)
            test_file = data_file
        if 'data_batch_' in fi:
            data_file = os.path.join(args.data_path, fi)
            train_files.append(data_file)

    # make train dataset
    match_train_path = os.path.join(args.out_path, 'match_train')
    if not os.path.exists(match_train_path):
        os.makedirs(match_train_path)
    class_ids = {}
    sample_count = 0
    for fi in train_files:
        with open(fi, 'rb') as fr:
            sp = pickle.load(fr, encoding='bytes')
        a = sp[b'data']
        b = sp[b'labels']
        for i in range(10000):
            lab = int(np.array(b[i]))
            if lab not in class_ids:
                class_ids[lab] = []
            class_ids[lab].append(sample_count)
            sample_count += 1
    # select class samples
    selected_train_ids = {}
    selected_eval_ids = []
    used_ids = {}
    for cls in class_ids.keys():
        train_ids = random.sample(class_ids[cls], args.train_samples)
        selected_train_ids[cls] = set(train_ids)
        used_ids[cls] = set(train_ids)
    for i in range(args.eval_files):
        eval_file_ids = []
        for cls in class_ids.keys():
            remain_ids = set(class_ids[cls]) - used_ids[cls]
            print('\teval file:', i, 'class:', cls, 'remained samples:', len(remain_ids))
            eval_ids = random.sample(list(remain_ids), args.eval_samples)
            eval_file_ids += eval_ids
            used_ids[cls] = used_ids[cls].union(set(eval_ids))
        selected_eval_ids.append(set(eval_file_ids))

    # dump samples
    dump_files = []
    for i in range(args.num_class):
        out_file = os.path.join(match_train_path, str(i) + '.pkl')
        fw = open(out_file, 'wb')
        dump_files.append(fw)
    sample_count = 0
    for fi in train_files:
        temp_class_ids = {}
        min_id = sample_count
        max_id = sample_count + 10000
        for cls in selected_train_ids.keys():
            temp_class_ids[cls] = set()
            for spid in selected_train_ids[cls]:
                if min_id <= spid < max_id:
                    temp_class_ids[cls].add(spid)
        with open(fi, 'rb') as fr:
            sp = pickle.load(fr, encoding='bytes')
        a = sp[b'data']
        b = sp[b'labels']
        for i in range(10000):
            img = a[i].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[i]).astype(np.long)
            cls = int(lab)
            if sample_count in temp_class_ids[cls]:
                pickle.dump([img, lab], dump_files[cls])
            sample_count += 1
    for di in dump_files:
        di.close()

    # make eval files
    eval_path = os.path.join(args.out_path, 'match_eval')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    for idf in range(args.eval_files):
        out_file = os.path.join(eval_path, str(idf) + '.pkl')
        with open(out_file, 'wb') as fw:
            sample_count = 0
            for fi in train_files:
                temp_ids = set()
                min_id = sample_count
                max_id = sample_count + 10000
                for spid in selected_eval_ids[idf]:
                    if min_id <= spid < max_id:
                        temp_ids.add(spid)
                with open(fi, 'rb') as fr:
                    sp = pickle.load(fr, encoding='bytes')
                a = sp[b'data']
                b = sp[b'labels']
                for i in range(10000):
                    img = a[i].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
                    lab = np.array(b[i]).astype(np.long)
                    if sample_count in temp_ids:
                        pickle.dump([img, lab], fw)
                    sample_count += 1

    # make tune eval and test files
    class_ids = {}
    with open(test_file, 'rb') as fr:
        sp = pickle.load(fr, encoding='bytes')
    a = sp[b'data']
    b = sp[b'labels']
    sample_count = 0
    for i in range(10000):
        lab = int(np.array(b[i]))
        if lab not in class_ids:
            class_ids[lab] = []
        class_ids[lab].append(sample_count)
        sample_count += 1
    dev_samples = {}
    test_samples = {}
    used_ids = {}
    for cls in class_ids.keys():
        dev_ids = random.sample(class_ids[cls], 500)
        dev_samples[cls] = dev_ids
        used_ids[cls] = set(dev_ids)
    for cls in class_ids.keys():
        remain_ids = set(class_ids[cls]) - used_ids[cls]
        test_samples[cls] = list(remain_ids)

    # make dump
    tune_eval_path = os.path.join(args.out_path, 'tune_eval')
    if not os.path.exists(tune_eval_path):
        os.makedirs(tune_eval_path)
    tune_test_path = os.path.join(args.out_path, 'tune_test')
    if not os.path.exists(tune_test_path):
        os.makedirs(tune_test_path)
    eval_dump_files = []
    for i in range(args.num_class):
        dump_file = os.path.join(tune_eval_path, str(i) + '.pkl')
        fw = open(dump_file, 'wb')
        eval_dump_files.append(fw)
    with open(test_file, 'rb') as fr:
        sp = pickle.load(fr, encoding='bytes')
    a = sp[b'data']
    b = sp[b'labels']
    for cls in dev_samples.keys():
        for spid in dev_samples[cls]:
            img = a[spid].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[spid]).astype(np.long)
            pickle.dump([img, lab], eval_dump_files[cls])
    for fi in eval_dump_files:
        fi.close()
    test_dump_files = []
    for i in range(args.num_class):
        dump_file = os.path.join(tune_test_path, str(i) + '.pkl')
        fw = open(dump_file, 'wb')
        test_dump_files.append(fw)
    with open(test_file, 'rb') as fr:
        sp = pickle.load(fr, encoding='bytes')
    a = sp[b'data']
    b = sp[b'labels']
    for cls in test_samples.keys():
        for spid in test_samples[cls]:
            img = a[spid].astype(np.float32).reshape([3, 32, 32]) / 255 - 0.5
            lab = np.array(b[spid]).astype(np.long)
            pickle.dump([img, lab], test_dump_files[cls])
    for fi in test_dump_files:
        fi.close()
