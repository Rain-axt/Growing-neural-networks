# -*-coding:utf8-*-

import pickle
import numpy as np
import os
import data_loader
import torch
import copy
import random


def read_additional_seq_ids_from_log(log_file, step):
    """
    read added sequence ids from log file
    :param log_file:
    :param step:
    :return:
    """
    seq_ids = {}
    with open(log_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    for idl, line in enumerate(lines):
        if '=== add branches done === step:' in line:
            cur_step = int(line.split(': ')[-1])
            if step == cur_step:
                target_line = lines[idl + 1]
                seq_list = target_line.split('{')[1].split('}')[0].split('], ')
                for si in seq_list:
                    key = int(si.split(': ')[0])
                    branches = []
                    for bi in si.split(': ')[1].split('[')[1].split(']')[0].split(', '):
                        branches.append(int(bi))
                    seq_ids[key] = branches
                break
    return seq_ids


def str2dic(target_line):
    """
    convert string to dict
    """
    seq_ids = {}
    seq_list = target_line.split('{')[1].split('}')[0].split('], ')
    for si in seq_list:
        key = int(si.split(': ')[0])
        branches = []
        for bi in si.split(': ')[1].split('[')[1].split(']')[0].split(', '):
            branches.append(int(bi))
        seq_ids[key] = branches
    return seq_ids


def read_prepared_branches_from_log(log_file):
    """
    read prepared branches from log file
    """
    seq_ids = {}
    with open(log_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    for line in lines:
        if line.startswith('\tin prepare:'):
            sid = int(line.split(' ')[3])
            branches = []
            dict_content = line.split('{')[1].split('}')[0].split(', ')
            for ci in dict_content:
                tokens = ci.split(': ')
                if tokens[1] == 'False':
                    branches.append(int(tokens[0]))
            seq_ids[sid] = branches
    return seq_ids


def get_average_class_accuracy(data_file, out_file, num_classes):
    """
    get average class accuracy
    :param data_file:
    :param out_file:
    :param num_classes:
    :return:
    """
    total_accuracy = np.zeros(shape=[num_classes], dtype=np.float32)
    with open(data_file, 'r', encoding='utf8') as fr:
        lines = fr.read().strip().split('\n')
    branch_count = 0
    for idl, line in enumerate(lines):
        if idl == 0:
            continue
        tokens = line.split('\t')
        if tokens[-1] == '':
            tokens = tokens[:-1]
        if len(tokens) < num_classes:
            continue
        # print(tokens)
        branch_count += 1
        recalls = []
        recall_sum = 0
        for idt, ti in enumerate(tokens):
            if idt < 2:
                continue
            recalls.append(float(ti))
            recall_sum += float(ti)
        other_accuracy = []
        for i in range(num_classes):
            other_accuracy.append((recall_sum - recalls[i]) / (num_classes - 1))
        for i in range(num_classes):
            total_accuracy[i] += ((recalls[i] + other_accuracy[i]) / 2)
    total_accuracy /= branch_count
    with open(out_file, 'w', encoding='utf8') as fw:
        for i in range(num_classes):
            fw.write(str(i) + '\t' + str(total_accuracy[i]) + '\n')


def build_data_loader(data_path, num_class, target_classes, repeat, batch_size):
    """
    build pytorch dataset
    :param data_path:
    :param num_class:
    :param target_classes:
    :param repeat:
    :param batch_size:
    :return:
    """
    dataset = data_loader.DataGeneratorClass(
        data_path=data_path, target_classes=target_classes, num_class=num_class, repeat=repeat)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    return dataloader


def build_data_loader_from_file(data_path, repeat, batch_size):
    """
    build data set from single file
    :param data_path:
    :param repeat:
    :param batch_size:
    :return:
    """
    dataset = data_loader.DataGeneratorImage(data_path=data_path, repeat=repeat, flatten=False)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    return dataloader


def build_data_loader_uniform(kernel_dim, batch_size):
    """
    build data load for random uniform [-0.5, 0.5)
    :param kernel_dim:
    :param batch_size:
    :return:
    """
    dataset = data_loader.DataGeneratorUniform(input_dim=kernel_dim)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
    return dataloader


def build_data_iter_shuffle(data_path, num_class):
    """
    build shuffled dataset
    :param data_path:
    :param num_class:
    :return:
    """
    data_objects = []
    for i in range(num_class):
        data_file = os.path.join(data_path, str(i) + '.pkl')
        fr = open(data_file, 'rb')
        data_objects.append(fr)
    closed_cls = set()
    cls_list = list(range(num_class))
    while len(closed_cls) < num_class:
        cls = random.choice(cls_list)
        if cls in closed_cls:
            continue
        try:
            sp = pickle.load(data_objects[cls])
            yield sp
        except EOFError:
            data_objects[cls].close()
            closed_cls.add(cls)
            if len(closed_cls) == num_class:
                break
            cls_list = []
            for i in range(num_class):
                if i not in closed_cls:
                    cls_list.append(i)


def merge_and_dump_train_data(data_path, num_class, batch_size, out_file):
    """
    merge and dump train data
    :param data_path:
    :param num_class:
    :param batch_size:
    :param out_file:
    :return:
    """
    data_iter = build_data_iter_shuffle(data_path=data_path, num_class=num_class)
    count = 0
    total_dic = {}
    total_labels = []
    total_fixed_output = []
    with open(out_file, 'wb') as fw:
        for sp in data_iter:
            dump_dic, fixed_output, dump_label = sp
            total_fixed_output.append(fixed_output)
            total_labels.append(dump_label)
            for tk in dump_dic.keys():
                if tk not in total_dic.keys():
                    total_dic[tk] = [dump_dic[tk]]
                else:
                    total_dic[tk].append(dump_dic[tk])
            count += 1
            if count != 0 and count % batch_size == 0:
                total_dump_dic = {}
                for tk in total_dic.keys():
                    total_dump_dic[tk] = np.concatenate(total_dic[tk], axis=0)
                dump_labels = np.concatenate(total_labels, axis=0)
                dump_fixed_output = np.concatenate(total_fixed_output, axis=0)
                pickle.dump([total_dump_dic, dump_fixed_output, dump_labels], fw)
                total_dic = {}
                total_labels = []
                total_fixed_output = []


def make_data_iteration_repeat(data_path, batch_size):
    """
    make iteration for training
    :param data_path:
    :param batch_size:
    :return:
    """
    with open(data_path, 'rb') as fr:
        while True:
            total_dic = {}
            all_labels = []
            total_fixed_out = []
            for i in range(batch_size):
                try:
                    dump_dic, fixed_out, dump_lab = pickle.load(fr)
                except EOFError:
                    fr.seek(0)
                    dump_dic, fixed_out, dump_lab = pickle.load(fr)
                all_labels.append(dump_lab)
                total_fixed_out.append(fixed_out)
                if i == 0:
                    for tk in dump_dic.keys():
                        total_dic[tk] = [dump_dic[tk]]
                else:
                    for tk in dump_dic.keys():
                        total_dic[tk].append(dump_dic[tk])
            all_labels = np.concatenate(all_labels, axis=0).astype(np.long)
            total_fixed_out = np.concatenate(total_fixed_out, axis=0)
            total_compute_dic = {}
            for tk in total_dic.keys():
                total_compute_dic[tk] = np.concatenate(total_dic[tk], axis=0)
            yield [total_compute_dic, total_fixed_out, all_labels]


def make_iteration(data_path):
    """
    make iteration for eval and test
    :param data_path:
    :return:
    """
    with open(data_path, 'rb') as fr:
        while True:
            try:
                dump_dic, fixed_out, dump_lab = pickle.load(fr)
                yield dump_dic, fixed_out, dump_lab
            except EOFError:
                break


def extract_number(in_str):
    """
    extract number from string
    """
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_str = ''
    for si in in_str:
        if (len(num_str) > 0) and (si not in number):
            break
        if si in number:
            num_str += si
    if len(num_str) == 0:
        return 0
    else:
        return int(num_str)


def get_target_threshold(input_file, target_count, non_target_count, min_precision, min_recall):
    """
    get the threshold, find the best threshold to split target and non-target data
    :param input_file:
    :param target_count:
    :param non_target_count:
    :param min_precision:
    :param min_recall:
    :return:
    """
    points = []
    max_value = None
    min_value = None
    with open(input_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                # class output, true-label, sample-id
                points.append([float(sp[0]), int(sp[1]), int(sp[2])])
                if max_value is None or max_value < sp[0]:
                    max_value = float(sp[0]) + 1e-3
                if min_value is None or min_value > sp[0]:
                    min_value = float(sp[0])
            except EOFError:
                break
    if len(points) == 0:
        return None
    print('\tin range samples:', len(points))
    # sort points and split to ranges
    sorted_points = sorted(points, key=lambda x: x[0])
    thresholds = []
    increment = (max_value - min_value) / 100
    start = min_value
    for i in range(100):
        thresholds.append(start)
        start += increment
    threshold_info = []
    for i in range(100):
        threshold_info.append([0, 0])
    cur_idx = 0
    for si in sorted_points:
        while cur_idx < 99:
            lower = thresholds[cur_idx]
            upper = thresholds[cur_idx + 1]
            if lower <= si[0] < upper:
                threshold_info[cur_idx][si[1]] += 1
                break
            else:
                cur_idx += 1
    for i in range(99, 0, -1):
        threshold_info[i - 1][0] += threshold_info[i][0]
        threshold_info[i - 1][1] += threshold_info[i][1]
    recall = 0
    threshold = min_value
    precision = 0
    non_target_rate = 0
    for i in range(100):
        if threshold_info[i][1] + threshold_info[i][0] == 0:
            continue
        non_target_rate = threshold_info[i][0] / non_target_count
        precision = threshold_info[i][1] / (threshold_info[i][1] + threshold_info[i][0])
        if precision > min_precision:
            threshold = thresholds[i]
            recall = threshold_info[i][1] / target_count
            break
    value_span = max_value - threshold
    if recall <= min_recall or value_span == 0:
        return None
    exclude_non_target_samples = set()
    selected_target_samples = set()
    mis_class_samples = set()
    for si in sorted_points:
        if si[1] == 0 and si[0] < threshold:
            exclude_non_target_samples.add(si[2])
        if si[0] >= threshold:
            if si[1] == 1:
                selected_target_samples.add(si[2])
            else:
                mis_class_samples.add(si[2])
    print('\tbest precision is:', precision, 'recall:', recall)
    print('\texclude samples:', 
          len(exclude_non_target_samples),
          target_count - len(selected_target_samples))
    profit = recall - non_target_rate
    return recall, precision, profit, threshold, value_span, \
        selected_target_samples, exclude_non_target_samples, mis_class_samples


def get_target_threshold_cmp(input_file,
                             target_count,
                             min_precision,
                             min_recall,
                             sample2weight):
    """
    get the threshold, find the best threshold to split target and non-target data
    :param input_file:
    :param target_count:
    :param min_precision:
    :param min_recall:
    :param sample2weight:
    :return:
    """
    points = []
    max_value = None
    min_value = None
    with open(input_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                # class output, true-label, sample-id
                points.append([float(sp[0]), int(sp[1]), int(sp[2])])
                if max_value is None or max_value < sp[0]:
                    max_value = float(sp[0]) + 1e-3
                if min_value is None or min_value > sp[0]:
                    min_value = float(sp[0])
            except EOFError:
                break
    if len(points) == 0:
        return None
    # sort points and split to ranges
    sorted_points = sorted(points, key=lambda x: x[0])
    thresholds = []
    increment = (max_value - min_value) / 100
    start = min_value
    for i in range(100):
        thresholds.append(start)
        start += increment
    threshold_info = []
    for i in range(100):
        threshold_info.append([set(), set()])
    cur_idx = 0
    for si in sorted_points:
        while cur_idx < 99:
            lower = thresholds[cur_idx]
            upper = thresholds[cur_idx + 1]
            if lower <= si[0] < upper:
                threshold_info[cur_idx][si[1]].add(si[2])
                break
            else:
                cur_idx += 1
        if cur_idx == 99:
            threshold_info[cur_idx][si[1]].add(si[2])
    recall = 0
    threshold = min_value
    precision = 0
    target_set = set()
    non_target_set = set()
    cur_pft = 0
    max_avg_pft = 0
    for i in range(99, -1, -1):
        tgt_set = threshold_info[i][1]
        for ti in tgt_set:
            cur_pft += sample2weight[ti]
        non_tgt_set = threshold_info[i][0]
        for nti in non_tgt_set:
            cur_pft += sample2weight[nti]
        target_set = target_set.union(tgt_set)
        non_target_set = non_target_set.union(non_tgt_set)
        if len(target_set) + len(non_target_set) == 0:
            continue
        cur_recall = len(target_set) / target_count
        cur_precision = len(target_set) / (len(target_set) + len(non_target_set))
        if cur_recall <= min_recall or cur_precision <= min_precision:
            continue
        avg_pft = cur_pft / (len(target_set) + len(non_target_set))
        if avg_pft > max_avg_pft:
            max_avg_pft = avg_pft
            threshold = thresholds[i]
            precision = cur_precision
            recall = cur_recall
    value_span = max_value - threshold
    if max_avg_pft == 0:
        return None
    exclude_non_target_samples = set()
    selected_target_samples = set()
    mis_class_samples = set()
    for si in sorted_points:
        if si[1] == 0 and si[0] < threshold:
            exclude_non_target_samples.add(si[2])
        if si[0] >= threshold:
            if si[1] == 1:
                selected_target_samples.add(si[2])
            else:
                mis_class_samples.add(si[2])
    return recall, precision, max_avg_pft, threshold, value_span, \
        selected_target_samples, exclude_non_target_samples, mis_class_samples


def get_target_threshold_cmp_eval(input_file,
                                  target_count,
                                  min_precision,
                                  min_recall,
                                  sample2weight,
                                  do_tuning=False):
    """
    get the threshold, find the best threshold to split target and non-target data
    :param input_file:
    :param target_count:
    :param min_precision:
    :param min_recall:
    :param sample2weight:
    :param do_tuning:
    :return:
    """
    points = []
    eval_points = []
    max_value = None
    min_value = None
    search_samples = len(sample2weight)
    with open(input_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                # class output, true-label, sample-id
                if int(sp[2]) < search_samples:
                    points.append([float(sp[0]), int(sp[1]), int(sp[2])])
                    if max_value is None or max_value < sp[0]:
                        max_value = float(sp[0])
                    if min_value is None or min_value > sp[0]:
                        min_value = float(sp[0])
                else:
                    eval_points.append([float(sp[0]), int(sp[1]), int(sp[2])])
            except EOFError:
                break
    if len(points) == 0:
        return None
    # sort points and split to ranges
    sorted_points = sorted(points, key=lambda x: x[0])
    thresholds = []
    min_value -= ((max_value - min_value) / 100)
    increment = (max_value - min_value) / 100
    start = min_value
    for i in range(100):
        thresholds.append(start)
        start += increment
    threshold_info = []
    for i in range(100):
        threshold_info.append([set(), set()])
    cur_idx = 0
    for si in sorted_points:
        while cur_idx < 99:
            lower = thresholds[cur_idx]
            upper = thresholds[cur_idx + 1]
            if lower < si[0] <= upper:
                threshold_info[cur_idx][si[1]].add(si[2])
                break
            else:
                cur_idx += 1
        if cur_idx == 99:
            threshold_info[cur_idx][si[1]].add(si[2])
    # make eval threshold info
    sorted_eval_points = sorted(eval_points, key=lambda x: x[0])
    eval_thd_info = []
    for i in range(100):
        eval_thd_info.append([0, 0])
    cur_idx = 0
    for si in sorted_eval_points:
        while cur_idx < 99:
            lower = thresholds[cur_idx]
            upper = thresholds[cur_idx + 1]
            if lower <= si[0] < upper:
                eval_thd_info[cur_idx][si[1]] += 1
                break
            else:
                cur_idx += 1
        if cur_idx == 99:
            eval_thd_info[cur_idx][si[1]] += 1
    for i in range(99, 0, -1):
        eval_thd_info[i - 1][0] = eval_thd_info[i - 1][0] + eval_thd_info[i][0]
        eval_thd_info[i - 1][1] = eval_thd_info[i - 1][1] + eval_thd_info[i][1]
    # do search
    recall = 0
    threshold = min_value
    precision = 0
    target_set = set()
    non_target_set = set()
    cur_pft = 0
    max_avg_pft = 0
    for i in range(99, -1, -1):
        tgt_set = threshold_info[i][1]
        for ti in tgt_set:
            cur_pft += sample2weight[ti]
        non_tgt_set = threshold_info[i][0]
        for nti in non_tgt_set:
            cur_pft += sample2weight[nti]
        target_set = target_set.union(tgt_set)
        non_target_set = non_target_set.union(non_tgt_set)
        if len(target_set) + len(non_target_set) == 0:
            continue
        cur_recall = len(target_set) / target_count
        cur_precision = len(target_set) / (len(target_set) + len(non_target_set) + 1e-10)
        cur_eval_precision = eval_thd_info[i][1] / (eval_thd_info[i][0] + eval_thd_info[i][1] + 1e-10)
        if cur_recall <= min_recall or cur_precision <= min_precision or cur_eval_precision <= min_precision:
            continue
        avg_pft = cur_pft / (len(target_set) + len(non_target_set) + 1e-10)
        if avg_pft > max_avg_pft:
            max_avg_pft = avg_pft
            threshold = thresholds[i]
            precision = cur_precision
            recall = cur_recall
    value_span = max_value - threshold
    if max_avg_pft == 0:
        return None
    selected_target_samples, mis_class_samples = process_sample_set(
        sorted_points=sorted_points,
        threshold=threshold,
        value_span=value_span,
        do_tuning=do_tuning
    )
    return recall, precision, max_avg_pft, threshold, value_span, \
        selected_target_samples, mis_class_samples


def process_sample_set(sorted_points, threshold, value_span, do_tuning):
    """
    post process of samples
    :param sorted_points:
    :param threshold:
    :param value_span:
    :param do_tuning:
    :return:
    """
    if not do_tuning:
        selected_target_samples = set()
        mis_class_samples = set()
        for si in sorted_points:
            if si[0] > threshold:
                if si[1] == 1:
                    selected_target_samples.add(si[2])
                else:
                    mis_class_samples.add(si[2])
        return selected_target_samples, mis_class_samples
    else:
        selected_target_samples = {}
        mis_class_samples = {}
        for si in sorted_points:
            if si[0] > threshold:
                normed_value = (si[0] - threshold) / (value_span + 1e-10)
                if si[1] == 1:
                    selected_target_samples[si[2]] = normed_value
                else:
                    mis_class_samples[si[2]] = normed_value
        return selected_target_samples, mis_class_samples
