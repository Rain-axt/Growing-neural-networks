# -*-coding:utf8-*-

import os
import pickle
from torch.utils.data import IterableDataset
import numpy as np
import random


class DataGenerator(IterableDataset):
    def __init__(self,
                 data_path,
                 repeat=True,
                 flatten=False,
                 transpose=False,
                 **kwargs):
        super(DataGenerator, self).__init__(**kwargs)
        self.data_path = data_path
        self.repeat = repeat
        self.flatten = flatten
        self.transpose = transpose

    def __iter__(self):
        with open(self.data_path, 'rb') as fr:
            while True:
                try:
                    sp = pickle.load(fr)
                    if not self.flatten:
                        if self.transpose:
                            yield np.transpose(sp[0], [2, 0, 1]).astype(np.float32), sp[1].astype(np.long)
                        else:
                            yield sp[0].astype(np.float32), sp[1].astype(np.long)
                    else:
                        yield np.reshape(sp[0], [-1]).astype(np.float32), sp[1].astype(np.long)
                except EOFError:
                    if self.repeat:
                        fr.seek(0)
                    else:
                        break


class DataGeneratorImage(IterableDataset):
    """
    data generator with loss tag
    """
    def __init__(self,
                 data_path,
                 repeat=True,
                 flatten=False,
                 **kwargs):
        super(DataGeneratorImage, self).__init__(**kwargs)
        self.data_path = data_path
        self.repeat = repeat
        self.flatten = flatten

    def __iter__(self):
        with open(self.data_path, 'rb') as fr:
            while True:
                try:
                    sp = pickle.load(fr)
                    # feature, label
                    if not self.flatten:
                        yield sp[0].astype(np.float32), sp[1].astype(np.long)
                    else:
                        yield np.reshape(sp[0], [-1]).astype(np.float32), sp[1].astype(np.long)
                except EOFError:
                    if self.repeat:
                        fr.seek(0)
                    else:
                        break


class DataGeneratorClass(IterableDataset):
    def __init__(self,
                 data_path,
                 num_class,
                 target_classes=None,
                 repeat=True,
                 **kwargs):
        super(DataGeneratorClass, self).__init__(**kwargs)
        self.data_path = data_path
        self.repeat = repeat
        self.num_class = num_class
        if target_classes is not None:
            if isinstance(target_classes, set):
                self.target_classes = target_classes
            else:
                self.target_classes = set(target_classes)
        else:
            self.target_classes = None
        self.cls2file = {}
        # build class map
        self.class_map = {}
        self.class_schedule = []
        if target_classes is not None:
            target_count = 1
            for i in range(self.num_class):
                if i in target_classes:
                    self.class_map[i] = np.array(target_count, dtype=np.long)
                    target_count += 1
                else:
                    self.class_map[i] = np.array(0, dtype=np.long)
            # make class schedule
            num_non_target_class = self.num_class - len(self.target_classes)
            for ti in self.target_classes:
                for i in range(num_non_target_class):
                    self.class_schedule.append(ti)
            for i in range(self.num_class):
                if i not in self.target_classes:
                    self.class_schedule.append(i)
        else:
            self.class_schedule = list(range(self.num_class))
        # build data loader
        self.open_files()

    def open_files(self):
        """
        open files
        :return:
        """
        for i in range(self.num_class):
            data_file = os.path.join(self.data_path, str(i) + '.pkl')
            fr = open(data_file, 'rb')
            self.cls2file[i] = fr

    def refresh_files(self):
        """
        put points to the initial of files
        """
        for i in range(self.num_class):
            self.cls2file[i].seek(0)

    def close_files(self):
        """
        close files
        :return:
        """
        for i in range(self.num_class):
            self.cls2file[i].close()

    def __iter__(self):
        if self.repeat:  # repeat only in training mode
            while True:
                cls = random.choice(self.class_schedule)
                fr = self.cls2file[cls]
                try:
                    sp = pickle.load(fr)
                    feat, lab = sp
                    if self.target_classes is None:
                        yield feat.astype(np.float32), lab.astype(np.long)
                    else:
                        lab = self.class_map[int(lab)]
                        yield feat.astype(np.float32), lab.astype(np.long)
                except EOFError:
                    fr.seek(0)
        else:
            for i in range(self.num_class):
                fr = self.cls2file[i]
                while True:
                    try:
                        sp = pickle.load(fr)
                        feat, lab = sp
                        if self.target_classes is None:
                            yield feat.astype(np.float32), lab.astype(np.long)
                        else:
                            lab = self.class_map[int(lab)]
                            yield feat.astype(np.float32), lab.astype(np.long)
                    except EOFError:
                        break


class DataGeneratorUniform(IterableDataset):
    """
    generate random uniform data of range [-0.5, 0.5)
    """
    def __init__(self, input_dim):
        super(DataGeneratorUniform, self).__init__()
        self.input_dim = input_dim

    def __iter__(self):
        while True:
            sample = np.random.rand(self.input_dim).astype(np.float32) - 0.5
            fake_label = np.array(0).astype(np.long)
            yield sample, fake_label
