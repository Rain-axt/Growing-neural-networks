# -*-coding:utf8-*-

import os
import numpy as np
import torch
import data_loader
import clustering_cm
import argparse
import pickle
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cluster')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--kernel_dim', type=int)
    parser.add_argument('--input_dims', type=str)
    parser.add_argument('--dim_span', type=int)
    parser.add_argument('--dim_shifts', type=str)
    parser.add_argument('--input_channel', type=int)
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    input_dims = [int(di) for di in args.input_dims.split(',')]
    dim_shifts = [int(si) for si in args.dim_shifts.split(',')]
    data_itor = data_loader.DataGeneratorClass(data_path=args.data_path, num_class=args.num_class, repeat=False)
    dim_segs = []
    start1 = 0
    start2 = 0
    dim_starts = set()
    while True:
        for i in range(args.input_channel):
            dim_range = [i, start1, start1 + args.dim_span, start2, start2 + args.dim_span]
            dim_segs.append(dim_range)
        start_token = str(start1) + '-' + str(start2)
        dim_starts.add(start_token)
        start2 += dim_shifts[1]
        if start2 + args.dim_span > input_dims[1]:
            cmp_start2 = input_dims[1] - args.dim_span
            start_token = str(start1) + '-' + str(cmp_start2)
            if start_token not in dim_starts:
                start2 = cmp_start2
            else:
                start1 += dim_shifts[0]
                start2 = 0
                if start1 + args.dim_span > input_dims[0]:
                    cmp_start1 = input_dims[0] - args.dim_span
                    start_token = str(cmp_start1) + '-' + str(start2)
                    if start_token not in dim_starts:
                        start1 = cmp_start1
                    else:
                        break
    # add random on dim ranges
    # random.shuffle(dim_segs)
    # dump data to out_path
    dump_path = os.path.join(args.out_path, 'ref_dumps')
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    out_files = []
    for i in range(len(dim_segs)):
        channel = str(dim_segs[i][0])
        start1 = str(dim_segs[i][1])
        start2 = str(dim_segs[i][3])
        basename = '_'.join([channel, start1, start2])
        out_file = os.path.join(dump_path, basename + '.pkl')
        out_files.append(out_file)
    dump_files = []
    seg_ids = []
    for idf, oi in enumerate(out_files):
        fw = open(oi, 'wb')
        dump_files.append(fw)
        seg_ids.append(idf)
        if idf % 200 == 0 or idf == len(out_files) - 1:
            print('dump data points:', idf)
            data_itor.refresh_files()
            sample_count = 0
            for data in data_itor:
                feat, lab = data
                lab = int(lab)
                for idx, sid in enumerate(seg_ids):
                    di = dim_segs[sid]
                    point = feat[di[0], di[1]:di[2], di[3]:di[4]]
                    point = np.reshape(point, [-1])
                    if point.size != args.kernel_dim:
                        print('\tsize error in file:', oi)
                        break
                    dump_point = [sample_count, point, lab]
                    pickle.dump(dump_point, dump_files[idx])
                sample_count += 1
            for di in dump_files:
                di.close()
            dump_files = []
            seg_ids = []
