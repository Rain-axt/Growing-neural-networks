# -*-coding:utf8-*-

import torch
import numpy as np
import os
import copy
import pickle
import random


class ParameterBlock(object):
    def __init__(self,
                 kernel_dim,
                 weight=None,
                 bias=None,
                 parameter_block=None,
                 **kwargs):
        super(ParameterBlock, self).__init__(**kwargs)
        if not parameter_block:
            self.kernel_dim = kernel_dim
            self.reference = 1
            # build layers
            self.linear = torch.nn.Linear(in_features=kernel_dim, out_features=kernel_dim, bias=True)
            if weight is not None:
                self.linear.weight = weight
            if bias is not None:
                self.linear.bias = bias
            # for matching
            self.in_upper = None
            self.in_lower = None
        else:
            self.copy_init(parameter_block=parameter_block)
        self.act = torch.nn.ReLU()
        self.require_grad = False
        self.is_cuda = False
        self.excluded = False
        self.train_state = False  # whether parameters of this block is in optimization list

    def copy_init(self, parameter_block):
        """
        copy initialization
        :param parameter_block:
        :return:
        """
        self.linear = parameter_block.linear
        self.reference = parameter_block.reference
        self.kernel_dim = parameter_block.kernel_dim
        if hasattr(parameter_block, 'in_upper'):
            self.in_upper = parameter_block.in_upper
        else:
            self.in_upper = None
        if hasattr(parameter_block, 'in_lower'):
            self.in_lower = parameter_block.in_lower
        else:
            self.in_lower = None

    def cuda(self, require_grad=True):
        self.require_grad = (require_grad and (not self.excluded))
        self.linear = self.linear.requires_grad_(requires_grad=require_grad).cuda()
        if self.in_upper is not None:
            self.in_upper = self.in_upper.cuda()
        if self.in_lower is not None:
            self.in_lower = self.in_lower.cuda()
        self.is_cuda = True

    def cpu(self, require_grad=True):
        self.require_grad = (require_grad and (not self.excluded))
        self.linear = self.linear.requires_grad_(requires_grad=require_grad).cpu()
        if self.in_upper is not None:
            self.in_upper = self.in_upper.cpu()
        if self.in_lower is not None:
            self.in_lower = self.in_lower.cpu()
        self.is_cuda = False

    def get_parameters(self):
        if (not self.require_grad) or self.train_state or self.excluded:
            return []
        else:
            self.train_state = True
            return [self.linear.weight, self.linear.bias]

    def __call__(self, inputs, set_bound=False):
        if set_bound:
            self.set_bound(inputs=inputs)
        linear_out = self.linear(inputs)
        act_out = self.act(linear_out)
        return act_out

    def bound_forward(self, inputs):
        cmp_res = None
        if self.in_upper is not None and self.in_lower is not None:
            cmp_upper = torch.any(torch.gt(inputs, self.in_upper), dim=1)
            cmp_lower = torch.any(torch.lt(inputs, self.in_lower), dim=1)
            cmp_res = torch.logical_or(cmp_upper, cmp_lower)
        linear_out = self.linear(inputs)
        act_out = self.act(linear_out)
        return act_out, cmp_res

    def add_reference(self):
        self.reference += 1

    def drop_block(self):
        self.reference -= 1
        return self.reference

    def exclude(self):
        self.excluded = True

    def undo_exclude(self):
        self.excluded = False

    def reset_train_state(self):
        self.train_state = False

    def set_bound(self, inputs):
        if self.is_cuda:
            inputs_tmp = torch.squeeze(inputs.clone().cpu().detach())
        else:
            inputs_tmp = torch.squeeze(inputs.clone().detach())
        if self.in_upper is None:
            self.in_upper = inputs_tmp
        else:
            self.in_upper = torch.maximum(self.in_upper, inputs_tmp)
        if self.in_lower is None:
            self.in_lower = inputs_tmp
        else:
            self.in_lower = torch.minimum(self.in_lower, inputs_tmp)

    def get_input_range(self):
        if self.in_upper is None or self.in_lower is None:
            return None
        mid = (self.in_upper + self.in_lower) / 2
        width = self.in_upper - self.in_lower
        return mid, width


class ClassMask(object):
    """
    used for class refinement, add mask to output of class
    each target class represent a single class
    """
    def __init__(self,
                 target_class,
                 num_class,
                 **kwargs):
        super(ClassMask, self).__init__(**kwargs)
        self.target_class = target_class
        self.num_classes = num_class
        # generate mask for inference
        class_mask = np.zeros([self.num_classes], dtype=np.float32)
        class_mask[target_class] = 1.0
        self.class_mask = torch.tensor(class_mask, dtype=torch.float32)
        self.scale = torch.nn.Parameter(torch.tensor(np.array(1), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(np.array(0), dtype=torch.float32))
        self.activation = torch.nn.ReLU()
        # support for mode selection
        self.train_mode = True
        self.available = True

    def set_scale(self, scale):
        self.scale = torch.nn.Parameter(torch.tensor(np.array(scale), dtype=torch.float32))

    def set_bias(self, bias):
        self.bias = torch.nn.Parameter(torch.tensor(np.array(bias), dtype=torch.float32))

    def cuda(self):
        self.class_mask = self.class_mask.cuda()
        self.scale = torch.nn.Parameter(self.scale.clone().detach().cuda())
        self.bias = torch.nn.Parameter(self.bias.clone().detach().cuda())

    def cpu(self):
        self.class_mask = self.class_mask.cpu()
        self.scale = torch.nn.Parameter(self.scale.clone().detach().cpu())
        self.bias = torch.nn.Parameter(self.bias.clone().detach().cpu())

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def enable(self):
        self.available = True

    def disable(self):
        self.available = False

    def get_parameter(self):
        if self.train_mode:
            return [self.scale, self.bias]
        else:
            return None

    def __call__(self, inputs):
        if self.available:
            selected_bias = (torch.gt(inputs, 0).float()) * self.activation(self.bias)
            output = (self.activation(self.scale) * inputs + selected_bias) * self.class_mask
        else:
            output = inputs
        return output

    def is_contributing(self):
        scale = np.squeeze(self.scale.clone().detach().numpy())
        bias = np.squeeze(self.bias.clone().detach().numpy())
        contribution = True
        if scale <= 0 and bias <= 0:
            contribution = False
        return contribution

    def get_target_class(self):
        return self.target_class


class LinearMask(object):
    def __init__(self):
        self.available = True
        self.train_mode = True
        self.scale = torch.nn.Parameter(torch.tensor(np.array(1), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(np.array(0), dtype=torch.float32))
        self.target_class = None

    def enable(self):
        self.available = True

    def disable(self):
        self.available = False

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def cuda(self):
        self.scale = torch.nn.Parameter(self.scale.clone().detach().cuda())
        self.bias = torch.nn.Parameter(self.bias.clone().detach().cuda())

    def cpu(self):
        self.scale = torch.nn.Parameter(self.scale.clone().detach().cpu())
        self.bias = torch.nn.Parameter(self.bias.clone().detach().cpu())

    def get_parameter(self):
        if self.train_mode:
            return [self.scale, self.bias]
        else:
            return None

    def __call__(self, inputs):
        if self.available:
            output = inputs * self.scale + self.bias
        else:
            output = inputs
        return output

    def get_target_class(self):
        return self.target_class


class ParameterTree(object):
    # part 1: initialization
    def __init__(self,
                 kernel_dim,
                 dim_range,
                 seq_id,
                 local_path,
                 num_class,
                 channel,
                 parameter_tree=None,
                 **kwargs):
        super(ParameterTree, self).__init__(**kwargs)
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.match_sample_path = os.path.join(self.local_path, 'matched_samples')
        if not os.path.exists(self.match_sample_path):
            os.makedirs(self.match_sample_path)
        self.kernel_dim = kernel_dim
        self.dim_range = dim_range
        self.sequence_id = seq_id
        self.num_class = num_class
        self.channel = channel

        # branch dic
        if not parameter_tree:
            self.branch_dic = {}
            self.joint_id = 0
            self.branch_id = 0
            self.joint_ref = {}
            self.start_to_id = {}
            self.max_start_layer = 0
            self.max_layer = 0
            self.start2id = {}
            # for matching support
            # key: branch_token, value: [profit, recall, precision, matched_class]
            self.matched_profit_info = {}
            self.trained_mask = set()
        else:
            self.copy_init(block_sequence=parameter_tree)

        # training support
        self.training_status = {}
        self.cuda_branches = set()
        self.flat_layer = torch.nn.Flatten()
        self.class_act = torch.nn.ReLU()

        # support for classifier exclusion
        self.excluded_classifier = set()

        # dump and cluster parameters
        self.dumps = {}  # used to store file object for dumping
        self.dump_ready = False
        self.dump_max_start = 0
        self.dump_blocks = set()
        self.dump_start2id = {}
        self.dumped_blocks = set()
        self.dumped_ref_blocks = set()
        self.dump_block2node = {}
        # for clusters set
        self.reference_clusters = set()
        self.target_clusters = set()  # store full file path

        # update clusters
        self.update_clusters()

    def copy_init(self, block_sequence):
        """
        copy init function
        :param block_sequence:
        :return:
        """
        self.sequence_id = block_sequence.sequence_id
        self.channel = block_sequence.channel
        self.branch_dic = block_sequence.branch_dic
        self.branch_id = block_sequence.branch_id
        self.start2id = block_sequence.start2id
        self.joint_ref = block_sequence.joint_ref
        self.joint_id = block_sequence.joint_id
        self.kernel_dim = block_sequence.kernel_dim
        self.num_class = block_sequence.num_class
        self.max_start_layer = block_sequence.max_start_layer
        self.max_layer = block_sequence.max_layer
        self.dim_range = block_sequence.dim_range
        if hasattr(block_sequence, 'matched_profit_info'):
            self.matched_profit_info = block_sequence.matched_profit_info
        else:
            self.matched_profit_info = {}
        if hasattr(block_sequence, 'trained_mask'):
            self.trained_mask = block_sequence.trained_mask
        else:
            self.trained_mask = set()

    def update_clusters(self):
        """
        update clusters and remove temp files
        """
        # remove target dumps
        dump_path = os.path.join(self.local_path, 'dumps')
        if os.path.exists(dump_path):
            dump_list = [os.path.join(dump_path, fi) for fi in os.listdir(dump_path)]
            for di in dump_list:
                if '+' in di:
                    continue
                os.remove(di)
        # remove temp cluster
        temp_cluster_path = os.path.join(self.local_path, 'temp_clusters')
        if os.path.exists(temp_cluster_path):
            temp_files = [os.path.join(temp_cluster_path, fi) for fi in os.listdir(temp_cluster_path)]
            for ti in temp_files:
                os.remove(ti)

        # load dumped clusters and reference clusters
        target_cluster_path = os.path.join(self.local_path, 'clusters')
        reference_cluster_path = os.path.join(self.local_path, 'ref_clusters')
        if os.path.exists(target_cluster_path):
            target_list = os.listdir(target_cluster_path)
            for ti in target_list:
                if '_act' in ti:
                    continue
                self.target_clusters.add(os.path.join(target_cluster_path, ti))
                # refresh dumped blocks
                self.dumped_blocks.add(ti.split('.')[0])
        if os.path.exists(reference_cluster_path):
            reference_list = os.listdir(reference_cluster_path)
            for ri in reference_list:
                if '+' in ri:
                    continue
                self.reference_clusters.add(os.path.join(reference_cluster_path, ri))
                # refresh dumped reference blocks
                self.dumped_ref_blocks.add(ri.split('.')[0])

    # 2. part 2: build layers
    def append_sequence(self, new_seq, layer, stem, detach=False):
        """
        append branches to sequence
        :param new_seq: a list, new block sequence of other block tree, or newly created
        :param layer: append start layer, here, the layer is the absolute layer
        :param stem: which branch the new sequence append on
        :param detach: whether to detach inputs of this branch
        :return: branch id for this block sequence
        """
        new_stem = stem
        if stem > -1 and layer == self.branch_dic[stem]['layer']:  # have same start layer
            new_stem = self.branch_dic[stem]['stem']
        classify_layer = torch.nn.Linear(in_features=self.kernel_dim, out_features=self.num_class, bias=True)
        seq_dic = {
            'sequence': copy.deepcopy(new_seq),
            'layer': layer,  # start absolute layer
            'classify_layer': classify_layer,
            'stem': new_stem,  # branch_id of the stem branch
            'joints': {},
            'detach': detach
        }
        if stem == -1:
            seq_dic['input_joint'] = -1
            seq_dic['detach'] = False
        else:
            if (layer - 1) in self.branch_dic[stem]['joints']:  # fetch output tensor from this joint
                # this branch compute from the output of the (layer - 1)th layer of stem
                seq_dic['input_joint'] = self.branch_dic[stem]['joints'][layer - 1]  # absolute layer
            else:
                # if there is no such joint, add joint to stem branch
                self.branch_dic[stem]['joints'][layer - 1] = self.joint_id  # absolute layer
                seq_dic['input_joint'] = self.joint_id
                self.joint_id += 1
            # update joint reference
            if seq_dic['input_joint'] in self.joint_ref:
                self.joint_ref[seq_dic['input_joint']][0] += 1
            else:
                self.joint_ref[seq_dic['input_joint']] = [1, stem, layer - 1]
            # update block set of this block tree
        self.branch_dic[self.branch_id] = seq_dic
        self.branch_id += 1

        if layer in self.start2id:
            self.start2id[layer].append(self.branch_id - 1)
        else:
            self.start2id[layer] = [self.branch_id - 1]

        if layer > self.max_start_layer:
            self.max_start_layer = layer
        if layer + len(new_seq) > self.max_layer:
            self.max_layer = layer + len(new_seq)
        # the returned branch id is used for training new branches only
        return self.branch_id - 1

    def add_class_mask(self,
                       branch_id,
                       classes,
                       weight=None,
                       bias=None):
        """
        add class mask for branch, by far, only support for two-class classification
        :param branch_id: branch id
        :param classes: a list of masked classes
        :param weight: initial value of weight of new classifier, (a numpy array)
        :param bias: initial value of bias of new classifier, (a numpy array)
        :return:
        """
        # create mask
        mask = ClassMask(target_class=classes[0], num_class=self.num_class)
        classify_layer = torch.nn.Linear(in_features=self.kernel_dim, out_features=len(classes), bias=True)
        if weight is not None:
            classify_layer.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        if bias is not None:
            classify_layer.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        # add mask
        self.branch_dic[branch_id]['class_mask'] = mask
        del self.branch_dic[branch_id]['classify_layer']
        self.branch_dic[branch_id]['classify_layer'] = classify_layer

    def exclude_classifier(self, branch_id):
        """
        exclude classifier for training
        :param branch_id:
        :return:
        """
        self.branch_dic[branch_id]['classify_layer'].eval()
        self.excluded_classifier.add(branch_id)

    def drop_branches(self,
                      branch_ids,
                      block_dic,
                      cuda=False):
        """
        drop branch and remove relative block
        no block sharing in the training phase
        :param branch_ids: a list of branch id to drop
        :param block_dic:
        :param cuda: whether the parameter is in cuda
        :return: None
        """
        bid2layer = [[bid, self.branch_dic[bid]['layer']] for bid in branch_ids]
        sorted_bid2layer = sorted(bid2layer, key=lambda x: x[1], reverse=True)
        # drop from higher layer
        for si in sorted_bid2layer:
            branch_id = si[0]
            # print('\tin drop:', self.sequence_id, branch_id)
            # remove match file
            matched_samples = os.path.join(self.match_sample_path, str(branch_id) + '.pkl')
            if os.path.exists(matched_samples):
                os.remove(matched_samples)
            branch = self.branch_dic[branch_id]
            # remove classify layers
            if cuda and branch_id in self.cuda_branches:
                self.branch_dic[branch_id]['classify_layer'] = self.branch_dic[branch_id]['classify_layer'].cpu()
            del self.branch_dic[branch_id]['classify_layer']
            # remove until the last joint
            block_sequence = branch['sequence']
            joints = branch['joints']
            start_layer = branch['layer']
            max_joints_layer = start_layer - 1
            if joints:
                max_joints_layer = max(list(joints.keys()))
            # remove blocks
            for i in range(len(block_sequence) - 1, - 1, -1):
                cur_layer = i + start_layer
                if max_joints_layer == cur_layer:
                    break
                # remove block
                ref_count = block_dic[block_sequence[i]].drop_block()
                if ref_count == 0:
                    del block_dic[block_sequence[i]]
                del block_sequence[i]
            # hand out rest blocks to other sub-branches
            # sort from the lowest layer
            sorted_joint_layers = sorted(list(joints.keys()))
            # joint-id to sub-branch-ids
            jid2bids = {}
            branch_joints = [joints[li] for li in joints.keys()]
            for bid in self.branch_dic.keys():
                jid = self.branch_dic[bid]['input_joint']
                if jid in branch_joints:
                    if jid not in jid2bids:
                        jid2bids[jid] = [bid]
                    else:
                        jid2bids[jid].append(bid)
            # refresh joint reward to find best-rewarded sub-branch
            joint_branches = []
            for li in sorted_joint_layers:
                jid = joints[li]
                # randomly select branch, not use reward
                best_branch = random.choice(jid2bids[jid])
                joint_branches.append([jid, best_branch, li])
            # change sub-branches
            cur_start_layer = start_layer
            cur_stem = self.branch_dic[branch_id]['stem']
            cur_input_joint = self.branch_dic[branch_id]['input_joint']
            last_joint = -1
            for ti in joint_branches:
                jid, bid, layer = ti
                # append block sequences
                blocks = block_sequence[cur_start_layer - start_layer: layer - start_layer + 1]
                self.branch_dic[bid]['sequence'] = blocks + self.branch_dic[bid]['sequence']
                # add joint and change stem
                self.branch_dic[bid]['joints'][layer] = jid
                self.branch_dic[bid]['stem'] = cur_stem
                # change branch layer and start2id
                self.branch_dic[bid]['layer'] = cur_start_layer
                if cur_start_layer == 0:
                    # set detach to False if start from input feature
                    self.branch_dic[bid]['detach'] = False
                # change input joint
                self.branch_dic[bid]['input_joint'] = cur_input_joint
                # change stem of other sub-branches of this joint
                for s_bid in jid2bids[jid]:
                    if s_bid == bid:
                        continue
                    self.branch_dic[s_bid]['stem'] = bid
                # change joint reference
                self.joint_ref[jid][1] = bid
                last_joint = jid  # the last joint need to decrease reference
                # refresh layer and stem
                cur_stem = bid
                cur_start_layer = layer + 1
                cur_input_joint = jid
            if last_joint != -1:
                self.joint_ref[last_joint][0] -= 1
                if self.joint_ref[last_joint][0] == 0:
                    stem = self.joint_ref[last_joint][1]
                    jl = self.joint_ref[last_joint][2]
                    # remove joint
                    del self.branch_dic[stem]['joints'][jl]
                    del self.joint_ref[last_joint]
            if not block_sequence:
                input_joint = self.branch_dic[branch_id]['input_joint']
                if input_joint != -1:
                    self.joint_ref[input_joint][0] -= 1
                    if self.joint_ref[input_joint][0] == 0:
                        stem = self.joint_ref[input_joint][1]
                        jl = self.joint_ref[input_joint][2]
                        del self.branch_dic[stem]['joints'][jl]
                        del self.joint_ref[input_joint]
                        # print('t\delete joint:', input_joint)
            # remove this branch
            del self.branch_dic[branch_id]

    def set_graph(self, branch_ids=None):
        """
        set start-to-id, to make least computation
        :param branch_ids:
        :return:
        """
        new_start2id = {}
        if branch_ids is None:
            for bid in self.branch_dic.keys():
                layer = self.branch_dic[bid]['layer']
                if layer not in new_start2id:
                    new_start2id[layer] = [bid]
                else:
                    new_start2id[layer].append(bid)
        else:
            for bid in branch_ids:
                layer = self.branch_dic[bid]['layer']
                if layer not in new_start2id:
                    new_start2id[layer] = [bid]
                else:
                    new_start2id[layer].append(bid)
            new_branches = set(branch_ids)
            all_branches = set(branch_ids)
            while len(new_branches) > 0:
                add_branches = set()
                for bid in new_branches:
                    stem = self.branch_dic[bid]['stem']
                    if stem == -1:
                        continue
                    if stem not in all_branches:
                        all_branches.add(stem)
                        add_branches.add(stem)
                        stem_layer = self.branch_dic[stem]['layer']
                        if stem_layer not in new_start2id:
                            new_start2id[stem_layer] = [stem]
                        else:
                            new_start2id[stem_layer].append(stem)
                new_branches = add_branches
                if len(new_branches) == 0:
                    break
        self.start2id = new_start2id

    # part 3: for training
    def prepare_training(self,
                         branch_ids,
                         block_dic,
                         include_branch,
                         trainable,
                         use_cuda=True):
        """
        prepare training
        :param branch_ids:
        :param block_dic:
        :param include_branch:
        :param trainable:
        :param use_cuda:
        :return:
        """
        parameters = []
        self.training_status = {}
        if not branch_ids:
            for bid in self.branch_dic.keys():
                self.training_status[bid] = True  # set to not trained
                parameters += self.branch_parameter(
                    branch_id=bid,
                    block_dic=block_dic,
                    use_cuda=use_cuda,
                    include_classifier=True,
                    include_branch=include_branch,
                    trainable=trainable
                )
        else:
            for bid in branch_ids:
                self.training_status[bid] = True
                parameters += self.branch_parameter(
                    branch_id=bid,
                    block_dic=block_dic,
                    use_cuda=use_cuda,
                    include_classifier=True,
                    include_branch=include_branch,
                    trainable=trainable
                )
            new_branches = len(branch_ids)
            # select stems for training branches
            while new_branches > 0:
                new_branches = 0
                to_add = set()
                for bid in self.training_status.keys():
                    if self.branch_dic[bid]['stem'] == -1:
                        continue
                    if self.branch_dic[bid]['stem'] not in self.training_status:
                        # for stems, just compute, no need to add parameters
                        # need to move this parameters to cuda if use cuda
                        to_add.add(self.branch_dic[bid]['stem'])
                        new_branches += 1
                for bid in to_add:
                    self.training_status[bid] = True
                    self.branch_parameter(
                        branch_id=bid,
                        block_dic=block_dic,
                        use_cuda=use_cuda,
                        include_classifier=False,
                        include_branch=include_branch,
                        trainable=False
                    )
        # re-build start2id
        new_start2id = {}
        for bid in self.training_status.keys():
            layer = self.branch_dic[bid]['layer']
            if layer not in new_start2id:
                new_start2id[layer] = [bid]
            else:
                new_start2id[layer].append(bid)
        self.start2id = new_start2id
        return parameters

    def branch_parameter(self,
                         branch_id,
                         block_dic,
                         use_cuda,
                         include_classifier,
                         include_branch,
                         trainable=True):
        """
        move branches to cuda
        :param branch_id:
        :param block_dic:
        :param use_cuda:
        :param include_classifier:
        :param include_branch:
        :param trainable:
        :return:
        """
        sequence = self.branch_dic[branch_id]['sequence']
        parameters = []
        for bli in sequence:  # if not train branches, the branched need to be in cuda to do computation
            if use_cuda:
                block_dic[bli].cuda(require_grad=trainable)
            if trainable and include_branch:
                parameters += block_dic[bli].get_parameters()
        if include_classifier:  # if not use classifier, no need to move classifier to cuda
            if use_cuda:
                if branch_id not in self.excluded_classifier:
                    self.branch_dic[branch_id]['classify_layer'] = \
                        self.branch_dic[branch_id]['classify_layer'].requires_grad_(requires_grad=trainable).cuda()
                else:
                    self.branch_dic[branch_id]['classify_layer'] = \
                        self.branch_dic[branch_id]['classify_layer'].requires_grad_(requires_grad=False).cuda()
                if 'class_mask' in self.branch_dic[branch_id]:
                    self.branch_dic[branch_id]['class_mask'].cuda()
                self.cuda_branches.add(branch_id)
            if trainable:
                if branch_id not in self.excluded_classifier:
                    parameters.append(self.branch_dic[branch_id]['classify_layer'].weight)
                    parameters.append(self.branch_dic[branch_id]['classify_layer'].bias)
                if 'class_mask' in self.branch_dic[branch_id]:
                    mask_param = self.branch_dic[branch_id]['class_mask'].get_parameter()
                    if mask_param is not None:
                        parameters += mask_param
        return parameters

    def enable_class_masks(self, branch_ids=None):
        """
        enable class masks
        :param branch_ids:
        :return:
        """
        if branch_ids is None:
            branch_ids = self.branch_dic.keys()
        for bid in branch_ids:
            if 'class_mask' in self.branch_dic[bid]:
                self.branch_dic[bid]['class_mask'].enable()
                self.branch_dic[bid]['class_mask'].eval()

    def clear_cuda(self):
        """
        clear cuda
        :return:
        """
        for bid in self.cuda_branches:
            self.branch_dic[bid]['classify_layer'] = \
                self.branch_dic[bid]['classify_layer'].requires_grad_(requires_grad=False).cpu()
            if 'class_mask' in self.branch_dic[bid]:
                self.branch_dic[bid]['class_mask'].cpu()
        self.cuda_branches.clear()

    def clear_training_status(self):
        self.training_status.clear()

    def save_classify_layer(self, branch_ids, step):
        """
        save branch classify layers
        :param branch_ids:
        :param step:
        :return:
        """
        if branch_ids is None:
            branch_ids = self.branch_dic.keys()
        save_file = os.path.join(self.local_path, 'checkpoint_' + str(step) + '.pkl')
        with open(save_file, 'wb') as fw:
            for bid in branch_ids:
                saved_layer = \
                    copy.deepcopy(self.branch_dic[bid]['classify_layer']).requires_grad_(requires_grad=False).cpu()
                token = str(self.sequence_id) + '_' + str(bid)
                save_list = [token, saved_layer]
                pickle.dump(save_list, fw)
        # save class mask
        class_mask_file = os.path.join(self.local_path, 'classmask_' + str(step) + '.pkl')
        with open(class_mask_file, 'wb') as fw:
            for bid in branch_ids:
                if 'class_mask' not in self.branch_dic[bid]:
                    continue
                save_mask = copy.deepcopy(self.branch_dic[bid]['class_mask'])
                save_mask.cpu()
                token = str(self.sequence_id) + '_' + str(bid)
                save_list = [token, save_mask]
                pickle.dump(save_list, fw)

    def load_classify_layers(self):
        """
        load classify layers and remove checkpoints
        :return:
        """
        loaded_branches = set()
        checkpoints = [fi for fi in os.listdir(self.local_path) if fi.startswith('checkpoint')]
        steps = []
        for ci in checkpoints:
            step = int(ci.split('.')[0].split('_')[1])
            steps.append(step)
        sorted_steps = sorted(steps, reverse=True)
        for si in sorted_steps:
            load_file = os.path.join(self.local_path, 'checkpoint' + '_' + str(si) + '.pkl')
            with open(load_file, 'rb') as fr:
                while True:  # load saved layer
                    try:
                        saved_layer = pickle.load(fr)
                        token, layer = saved_layer
                        sid, bid = token.split('_')
                        bid = int(bid)
                        if bid not in loaded_branches and bid in self.branch_dic:
                            self.branch_dic[bid]['classify_layer'] = layer
                            loaded_branches.add(bid)
                    except EOFError:
                        break
            os.remove(load_file)
        # load class mask
        loaded_masks = set()
        class_mask_files = [fi for fi in os.listdir(self.local_path) if fi.startswith('classmask')]
        steps = []
        for ci in class_mask_files:
            step = int(ci.split('.')[0].split('_')[1])
            steps.append(step)
        sorted_steps = sorted(steps, reverse=True)
        for si in sorted_steps:
            load_file = os.path.join(self.local_path, 'classmask' + '_' + str(si) + '.pkl')
            with open(load_file, 'rb') as fr:
                while True:  # load saved class mask
                    try:
                        save_mask = pickle.load(fr)
                        token, mask = save_mask
                        sid, bid = token.split('_')
                        bid = int(bid)
                        if bid not in loaded_masks and bid in self.branch_dic:
                            self.branch_dic[bid]['class_mask'] = mask
                            loaded_masks.add(bid)
                    except EOFError:
                        break
            os.remove(load_file)

    # part 4: forward computation
    def forward(self,
                block_dic,
                inputs,
                branch_ids=None,
                mode='infer',
                norm='relu',
                set_bound=False,
                use_bound=False):
        """
        :param block_dic: block dic maintained by reinforcement controller, used to do forward propagation
        :param inputs: inputs of shape: [batch_size, input_dim]
        :param mode: train or infer, in the inference mode, the output is record to count reward
        :param branch_ids: in 'inference mode' branch ids to record output
                           in 'training mode' branch ids to compute classification
        :param norm: normalization type for output to attention
        :param set_bound: if True, set input boundary of parameter blocks
        :param use_bound: if True, use boundary for parameter
        :return: prediction of this block_tree
        """
        clipped_inputs = self.flat_layer(
            inputs[:, self.channel, self.dim_range[0]:self.dim_range[1], self.dim_range[2]:self.dim_range[3]])
        # build a temporary dic here to store {joint_id: joint_tensor}, and use this dic to do further computation
        joint_dic = {}
        forward_out = {}
        for i in range(self.max_start_layer + 1):
            if i not in self.start2id:
                continue
            for bid in self.start2id[i]:
                if i == 0:
                    temp_tensor = clipped_inputs
                else:
                    if self.branch_dic[bid]['input_joint'] in joint_dic:
                        temp_tensor = joint_dic[self.branch_dic[bid]['input_joint']]
                    else:
                        continue
                branch = self.branch_dic[bid]
                start_layer = branch['layer']
                joints = branch['joints']
                cmp_res = None
                temp_out = temp_tensor
                if branch['detach'] and mode == 'train':  # only in train mode, use detach
                    temp_out = temp_out.detach()
                for layer, block_id in enumerate(branch['sequence']):
                    absolute_layer = start_layer + layer
                    if layer == 0 and use_bound:
                        temp_out, cmp_res = block_dic[block_id].bound_forward(temp_out)
                    else:
                        temp_out = block_dic[block_id](temp_out, set_bound=set_bound)
                    if absolute_layer in joints:
                        joint_dic[joints[absolute_layer]] = temp_out
                if branch_ids is None or bid in branch_ids:
                    out_classification = branch['classify_layer'](temp_out)
                    if norm == 'l2':  # with joint training and balanced dataset
                        out_norm = torch.norm(out_classification, p=2, dim=1)
                        out_norm = torch.unsqueeze(out_norm, dim=1)
                        branch_classification = out_classification / (out_norm + 1e-10)
                    elif norm == 'relu':
                        branch_classification = self.class_act(out_classification)
                    else:
                        branch_classification = out_classification
                    token = str(self.sequence_id) + '_' + str(bid)
                    if use_bound:
                        forward_out[token] = [branch_classification, cmp_res]
                    else:
                        forward_out[token] = branch_classification
        return forward_out

    # part 5: information retrieval
    def trace_blocks(self, branch_id, target_block):
        """
        trace blocks to target-block
        :param branch_id:
        :param target_block:
        :return:
        """
        cur_branch = branch_id
        cur_start_layer = self.branch_dic[branch_id]['layer']
        target_layer = 0
        block_sequence = []  # put blocks in order
        seq_len = len(self.branch_dic[cur_branch]['sequence'])
        while True:
            sequence = self.branch_dic[cur_branch]['sequence'][:seq_len]
            if target_block in sequence:
                target_index = sequence.index(target_block)
                target_layer = cur_start_layer + target_index
                block_sequence = sequence[target_index:] + block_sequence
                break
            else:
                block_sequence = sequence[:seq_len] + block_sequence
            stem = self.branch_dic[cur_branch]['stem']
            if stem == -1:
                break
            cur_branch = stem
            last_layer = cur_start_layer
            cur_start_layer = self.branch_dic[cur_branch]['layer']
            seq_len = last_layer - cur_start_layer
        return block_sequence, target_layer, cur_branch

    # part 6: for dump
    def trace_branch(self, branch_id):
        """
        trace branch to root
        :param branch_id:
        :return:
        """
        branches = set()
        branches.add(branch_id)
        cur_bid = branch_id
        while True:
            stem = self.branch_dic[cur_bid]['stem']
            if stem == -1:
                break
            cur_bid = stem
            branches.add(cur_bid)
        return branches

    def trace_and_select(self, branch_ids, select=0, ref=False):
        """
        branch ids to compute and run
        :param branch_ids:
        :param select:
        :param ref:
        :return:
        """
        # do branch tracing
        total_blocks = []
        total_branches = set()
        for bid in branch_ids:
            traced_branches = self.trace_branch(branch_id=bid)
            for tbi in traced_branches:
                for idb, bli in enumerate(self.branch_dic[tbi]['sequence']):
                    # discriminate blocks of different branches, there splitting is different
                    block_name = str(tbi) + '_' + str(bli)
                    node = str(bid) + '_' + str(bli)
                    if not ref:
                        if block_name not in self.dump_block2node:
                            self.dump_block2node[block_name] = [node]
                        else:
                            self.dump_block2node[block_name].append(node)
                        total_blocks.append(node)
                    else:
                        self.dump_block2node[block_name] = [block_name]
                        total_blocks.append(block_name)
            total_branches = total_branches.union(traced_branches)
        if select > 0:
            if select > len(total_blocks):
                select = len(total_blocks)
            total_blocks = random.sample(total_blocks, select)
        total_blocks = set(total_blocks)  # remove redundant blocks
        # put branches in order
        max_start = 0
        start2id = {}
        for bi in total_branches:
            start_layer = self.branch_dic[bi]['layer']
            if start_layer > max_start:
                max_start = start_layer
            if start_layer in start2id:
                start2id[start_layer].append(bi)
            else:
                start2id[start_layer] = [bi]
        self.dump_max_start = max_start
        self.dump_blocks = total_blocks
        self.dump_start2id = start2id

    def prepare_dumps(self, branch_ids, temp=False, select=0):
        """
        make file objects here
        :param branch_ids:
        :param temp:
        :param select:
        :return:
        """
        if not temp:
            data_path = os.path.join(self.local_path, 'dumps')
            cluster_path = os.path.join(self.local_path, 'clusters')
        else:
            data_path = os.path.join(self.local_path, 'ref_dumps')
            cluster_path = os.path.join(self.local_path, 'ref_clusters')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        self.trace_and_select(branch_ids=branch_ids, select=select, ref=temp)
        out_files = set()
        for di in self.dump_blocks:
            outfile = os.path.join(data_path, di + '.pkl')
            if temp:
                if di in self.dumped_ref_blocks:
                    continue
                else:
                    self.dumped_ref_blocks.add(di)
            else:
                if di in self.dumped_blocks:  # do not need to dump exist node
                    continue
                else:
                    self.dumped_blocks.add(di)
            out_files.add(outfile)
            fw = open(outfile, 'wb')
            self.dumps[di] = fw
        self.dump_ready = True
        return out_files

    def delete_dumps(self):
        """
        clear dump-related parameters here
        :return:
        """
        for k in self.dumps.keys():
            self.dumps[k].close()
        self.dumps = {}
        self.dump_max_start = 0
        self.dump_blocks = set()
        self.dump_start2id = {}
        self.dump_ready = False
        self.dump_block2node = {}

    def ref_forward_dump(self,
                         inputs,
                         block_dic,
                         sample_id,
                         label):
        """
        dump output of specific node
        :param inputs:
        :param block_dic:
        :param sample_id:
        :param label:
        :return:
        """
        assert self.dump_ready
        clipped_inputs = self.flat_layer(
            inputs[:, self.channel, self.dim_range[0]:self.dim_range[1], self.dim_range[2]:self.dim_range[3]])
        # build a temporary dic here to store {joint_id: joint_tensor}, and use this dic to do further computation
        block_points = {}
        bid2label = {}
        joint_dic = {}
        for i in range(self.dump_max_start + 1):
            if i not in self.dump_start2id:
                continue
            for bid in self.dump_start2id[i]:
                if i == 0:
                    temp_tensor = clipped_inputs
                else:
                    temp_tensor = joint_dic[self.branch_dic[bid]['input_joint']]
                branch = self.branch_dic[bid]
                joints = branch['joints']
                start_layer = branch['layer']
                temp_out = temp_tensor
                run_blocks = branch['sequence']
                for bli in range(len(run_blocks)):
                    layer = bli + start_layer
                    block_name = str(bid) + '_' + str(run_blocks[bli])
                    temp_out = block_dic[run_blocks[bli]](temp_out)
                    if block_name in self.dump_block2node:
                        block_points[block_name] = np.squeeze(temp_out.clone().detach().numpy())
                    if layer in joints:
                        joint_dic[joints[layer]] = temp_out
                bid2label[bid] = label
        for block_name in block_points.keys():
            temp_point = block_points[block_name]
            for node in self.dump_block2node[block_name]:
                bid = int(node.split('_')[0])
                branch_label = bid2label[bid]
                if node in self.dumps:  # dump nodes
                    point = [sample_id, temp_point, branch_label]
                    pickle.dump(point, self.dumps[node])

    def target_forward_dump(self,
                            inputs,
                            block_dic,
                            sample_id,
                            norm='none'):
        """
        dump output of specific node
        :param inputs:
        :param block_dic:
        :param sample_id:
        :param norm:
        :return:
        """
        assert self.dump_ready
        clipped_inputs = inputs
        # build a temporary dic here to store {joint_id: joint_tensor}, and use this dic to do further computation
        block_points = {}
        node2value = {}
        for i in range(self.dump_max_start + 1):
            if i not in self.dump_start2id:
                continue
            for bid in self.dump_start2id[i]:
                branch = self.branch_dic[bid]
                block_sequence = branch['sequence']
                for li in range(len(block_sequence)):
                    run_blocks = block_sequence[li:]
                    target_node = str(bid) + '_' + str(run_blocks[0])
                    mid, width = block_dic[run_blocks[0]].get_input_range()
                    temp_out = (clipped_inputs + mid) * width
                    # dump inputs of target block
                    if target_node in self.dump_block2node:
                        block_points[target_node] = np.squeeze(temp_out.clone().detach().numpy())
                    for bli in range(len(run_blocks)):
                        temp_out = block_dic[run_blocks[bli]](temp_out)
                    out_classification = branch['classify_layer'](temp_out)
                    if norm == 'l2':  # with joint training and balanced dataset
                        out_norm = torch.norm(out_classification, p=2, dim=1)
                        out_norm = torch.unsqueeze(out_norm, dim=1)
                        branch_classification = out_classification / (out_norm + 1e-10)
                    elif norm == 'relu':
                        branch_classification = self.class_act(out_classification)
                    else:
                        branch_classification = out_classification
                    class_out_numpy = np.squeeze(branch_classification.detach().numpy())
                    node2value[target_node] = class_out_numpy
        for block_name in block_points.keys():
            temp_point = block_points[block_name]
            for node in self.dump_block2node[block_name]:
                if node in self.dumps:  # dump nodes
                    for branch_label in range(self.num_class):
                        point = [sample_id, temp_point, branch_label, node2value[node][branch_label]]
                        pickle.dump(point, self.dumps[node])

    # part 7: for matching
    def dump_matched_samples(self,
                             branch_id,
                             threshold,
                             value_span,
                             recall,
                             precision,
                             profit,
                             matched_class,
                             target_samples,
                             mis_class_samples):
        """
        dump matched samples
        :param branch_id:
        :param threshold:
        :param value_span:
        :param recall:
        :param precision:
        :param profit:
        :param matched_class:
        :param target_samples:
        :param mis_class_samples:
        :return:
        """
        self.branch_dic[branch_id]['out_thd'] = threshold
        self.branch_dic[branch_id]['val_span'] = value_span
        branch_token = str(self.sequence_id) + '_' + str(branch_id)
        self.matched_profit_info[branch_token] = [profit, recall, precision, matched_class]
        dump_file = os.path.join(self.match_sample_path, str(branch_id) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(matched_class, fw)
            pickle.dump(target_samples, fw)
            pickle.dump(mis_class_samples, fw)

    def load_matched_samples(self, branch_id):
        """
        load matched samples
        :param branch_id:
        :return:
        """
        dump_file = os.path.join(self.match_sample_path, str(branch_id) + '.pkl')
        with open(dump_file, 'rb') as fr:
            matched_class = pickle.load(fr)
            target_samples = pickle.load(fr)
            mis_class_samples = pickle.load(fr)
        branch_token = str(self.sequence_id) + '_' + str(branch_id)
        profit = self.matched_profit_info[branch_token][0]
        return matched_class, target_samples, mis_class_samples, profit

    # part 8: for tuning
    def prepare_branch_out(self, branch_ids):
        """
        prepare dataset file
        :return:
        """
        self.set_graph(branch_ids=branch_ids)
        for bid in branch_ids:
            if 'class_mask' in self.branch_dic[bid]:
                self.branch_dic[bid]['class_mask'].eval()

    def dump_branch_out(self, inputs, branch_ids, block_dic, norm='relu'):
        """
        dump branch out to file
        :param inputs:
        :param branch_ids:
        :param block_dic:
        :param norm:
        :return:
        """
        batch_size = int(inputs.shape[0])
        clipped_inputs = self.flat_layer(
            inputs[:, self.channel, self.dim_range[0]:self.dim_range[1], self.dim_range[2]:self.dim_range[3]])
        # build a temporary dic here to store {joint_id: joint_tensor}, and use this dic to do further computation
        joint_dic = {}
        dump_dic = {}
        for i in range(self.max_start_layer + 1):
            if i not in self.start2id:
                continue
            for bid in self.start2id[i]:
                if i == 0:
                    temp_tensor = clipped_inputs
                else:
                    temp_tensor = joint_dic[self.branch_dic[bid]['input_joint']]
                branch = self.branch_dic[bid]
                start_layer = branch['layer']
                joints = branch['joints']
                temp_out = temp_tensor
                cmp_res = None
                for layer, block_id in enumerate(branch['sequence']):
                    absolute_layer = start_layer + layer
                    if layer == 0 and 'out_thd' in branch:
                        temp_out, cmp_res = block_dic[block_id].bound_forward(temp_out)
                    else:
                        temp_out = block_dic[block_id](temp_out, set_bound=False)
                    if absolute_layer in joints:
                        joint_dic[joints[absolute_layer]] = temp_out
                token = str(self.sequence_id) + '_' + str(bid)
                if branch_ids is None or bid in branch_ids:
                    if 'out_thd' in branch:
                        out_classification = branch['classify_layer'](temp_out)
                        branch_out = out_classification.clone().detach().numpy()
                        branch_out = (branch_out - self.branch_dic[bid]['out_thd']) \
                            / (self.branch_dic[bid]['val_span'] + 1e-10)
                        branch_out = np.maximum(branch_out, 0)
                        cmp_res = cmp_res.clone().detach().numpy()
                        final_result = []
                        for bi in range(batch_size):
                            if not bool(cmp_res[bi]):
                                if bid in self.trained_mask:
                                    branch_classification = self.branch_dic[bid]['class_mask'](
                                        torch.tensor(branch_out[bi:bi + 1, :], dtype=torch.float32))
                                    final_result.append(branch_classification.clone().detach().numpy())
                                else:
                                    final_result.append(branch_out[bi:bi + 1, :])
                            else:
                                if bid in self.trained_mask:
                                    branch_classification = self.branch_dic[bid]['class_mask'](
                                        torch.tensor(np.zeros([1, 1], dtype=np.float32), dtype=torch.float32)
                                    )
                                    final_result.append(branch_classification.clone().detach().numpy())
                                else:
                                    final_result.append(np.zeros([1, 1], dtype=np.float32))
                        dump_dic[token] = np.concatenate(final_result, axis=0)
                    else:
                        out_classification = branch['classify_layer'](temp_out)
                        if norm == 'l2':  # with joint training and balanced dataset
                            out_norm = torch.norm(out_classification, p=2, dim=1)
                            out_norm = torch.unsqueeze(out_norm, dim=1)
                            branch_classification = out_classification / (out_norm + 1e-10)
                        elif norm == 'relu':
                            branch_classification = self.class_act(out_classification)
                        else:
                            branch_classification = out_classification
                        dump_dic[token] = branch_classification.clone().detach().numpy()
        return dump_dic

    def prepare_tuning(self, branch_ids, use_cuda=True):
        """
        open data set and return parameter list
        :param branch_ids:
        :param use_cuda:
        :return:
        """
        parameters = []
        train_branch_ids = []
        for bid in branch_ids:
            if 'class_mask' in self.branch_dic[bid]:
                if bid in self.trained_mask:
                    self.branch_dic[bid]['class_mask'].eval()
                    continue
                train_branch_ids.append(bid)
                self.branch_dic[bid]['class_mask'].train()
                self.branch_dic[bid]['class_mask'].enable()
                if use_cuda:
                    self.branch_dic[bid]['class_mask'].cuda()
                mask_parameter = self.branch_dic[bid]['class_mask'].get_parameter()
                if mask_parameter is not None:
                    parameters += mask_parameter
        # set computation graph
        self.set_graph(branch_ids=train_branch_ids)
        return parameters, train_branch_ids

    def tuning_forward(self, branch_ids, token2data, use_cuda):
        """
        do forward computation for tuning
        :param branch_ids:
        :param token2data:
        :param use_cuda:
        :return:
        """
        forward_out = {}
        for bid in branch_ids:
            token = str(self.sequence_id) + '_' + str(bid)
            inputs = torch.tensor(token2data[token], dtype=torch.float32)
            if use_cuda:
                inputs = inputs.cuda()
            if 'class_mask' in self.branch_dic[bid]:
                outputs = self.branch_dic[bid]['class_mask'](inputs)
            else:
                outputs = inputs
            forward_out[token] = outputs
        return forward_out

    def finish_tuning(self, branch_ids, use_cuda=True):
        """
        finish tuning
        :param branch_ids:
        :param use_cuda:
        :return:
        """
        for bid in branch_ids:
            if 'class_mask' in self.branch_dic[bid]:
                if use_cuda:
                    self.branch_dic[bid]['class_mask'].cpu()
            else:
                if use_cuda:
                    self.branch_dic[bid]['classify_layer'] = self.branch_dic[bid]['classify_layer'].cpu()
                self.branch_dic[bid]['classify_layer'] = \
                    self.branch_dic[bid]['classify_layer'].requires_grad_(requires_grad=False)
        # load parameter here
        print('\tload parameters:', self.sequence_id)
        self.load_classify_layers()
        drop_branches = []
        for bid in branch_ids:
            if 'class_mask' in self.branch_dic[bid]:
                self.branch_dic[bid]['class_mask'].eval()
                if not self.branch_dic[bid]['class_mask'].is_contributing():
                    drop_branches.append(bid)
                else:
                    if bid not in self.trained_mask:
                        self.trained_mask.add(bid)
                # remove dumped samples
                dump_file = os.path.join(self.match_sample_path, str(bid) + '.pkl')
                if os.path.exists(dump_file):
                    os.remove(dump_file)
        return drop_branches

    def clear_trained_mask(self, branch_ids):
        """
        :return:
        """
        if branch_ids is None:
            self.trained_mask.clear()
        else:
            for bid in branch_ids:
                self.trained_mask.remove(bid)

    def set_trained_mask(self):
        """
        add all matched branches to trained mask
        :return: 
        """
        for bid in self.branch_dic.keys():
            self.trained_mask.add(bid)

    # part x: other temp functions
    def reload_class_mask(self):
        for bid in self.branch_dic.keys():
            if 'class_mask' in self.branch_dic[bid]:
                target_class = self.branch_dic[bid]['class_mask'].get_target_class()
                del self.branch_dic[bid]['class_mask']
                new_class_mask = ClassMask(
                    target_class=target_class, num_class=self.num_class)
                self.branch_dic[bid]['class_mask'] = new_class_mask

    def reload_classifier(self, branch_dic):
        for bid in branch_dic.keys():
            self.branch_dic[bid]['classify_layer'] = branch_dic[bid]['classify_layer']


def parameter_transfer_numpy(kernel, bias, trans_dim_order, trans_dim_scale, trans_dim_shift, target_center):
    """
    used for initialize new parameter block
    :param kernel: each row is a single kernel
    :param bias: bias of original linear layer
    :param trans_dim_order: index -> value, target_dim -> reference dim
    :param trans_dim_scale:
    :param trans_dim_shift:
    :param target_center:
    :return: kernels applied to reference node
    """
    kernel_dim = trans_dim_order.shape[0]
    # check if there is 0 in scale
    bad_res = False
    for i in range(kernel_dim):
        if trans_dim_scale[0, i] == 0:
            bad_res = True
            break
    if bad_res:
        print('\tthere is INF in new kernels')
        return None
    if np.any(np.isinf(trans_dim_shift)) or np.any(np.isnan(trans_dim_shift)):
        print('\tthere is INF or NAN in trans dim shift')
        return None
    if np.any(np.isinf(target_center)) or np.any(np.isnan(target_center)):
        print('\tthere is INF or NAN in target center')
        return None
    if np.any(np.isinf(trans_dim_scale)) or np.any(np.isnan(trans_dim_scale)):
        print('\tthere is INF or NAN in trans dim scale', trans_dim_scale)
        return None
    new_kernels = []
    new_bias = copy.deepcopy(bias)
    # for debug
    for idk in range(kernel.shape[0]):
        new_kernel = np.zeros([1, kernel.shape[1]], dtype=np.float32) + kernel[idk, :]
        scaled_kernel = new_kernel / trans_dim_scale
        trans_kernel = scaled_kernel[:, trans_dim_order]
        bias_offset = np.sum(np.squeeze(new_kernel) * np.squeeze(target_center)) - \
            np.sum(np.squeeze(scaled_kernel) * trans_dim_shift)
        new_bias[idk] += bias_offset
        new_kernels.append(trans_kernel)
    if np.any(np.isnan(new_bias)) or np.any(np.isinf(new_bias)):
        print('\tthere is NAN or INF in new bias', new_bias)
        return None
    for ki in new_kernels:
        if np.any(np.isnan(ki)) or np.any(np.isinf(ki)):
            print('\tthere is NAN or INF in new kernels')
            bad_res = True
            break
    if bad_res:
        return None
    new_weight = torch.nn.Parameter(torch.tensor(np.concatenate(new_kernels, axis=0), dtype=torch.float32))
    new_bias = torch.nn.Parameter(torch.tensor(new_bias, dtype=torch.float32))
    return new_weight, new_bias


def select_best_samples(input_file, min_samples, sample_rate=0.2):
    """
    select best branch-class samples to dump
    :param input_file:
    :param min_samples:
    :param sample_rate:
    :return:
    """
    # load and sort points
    class_samples = {}  # class to list of samples
    with open(input_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                label = int(sp[2])
                if label not in class_samples:
                    class_samples[label] = [sp]
                else:
                    class_samples[label].append(sp)
            except EOFError:
                break
    class_sorted_samples = {}
    for cli in class_samples.keys():
        if len(class_samples[cli]) < min_samples:
            continue
        sorted_samples = sorted(class_samples[cli], key=lambda x: x[3], reverse=True)
        class_sorted_samples[cli] = sorted_samples
    # dump sorted points
    dump_file = os.path.join(os.path.dirname(input_file), 'new_points.pkl')
    class_value_range = {}
    class_act_range = {}
    with open(dump_file, 'wb') as fw:
        for cli in class_sorted_samples.keys():
            to_dump = int(len(class_sorted_samples[cli]) * sample_rate)
            class_value_range[cli] = [class_sorted_samples[cli][to_dump - 1], class_sorted_samples[cli][0]]
            values = []
            for i in range(to_dump):
                point = class_sorted_samples[cli][i]
                dump_point = [point[0], point[1], point[2], point[3]]
                pickle.dump(dump_point, fw)
                values.append(point[1])
            min_act_value = np.min(np.array(values), axis=0)
            max_act_value = np.max(np.array(values), axis=0)
            class_act_range[cli] = [min_act_value, max_act_value]
    os.remove(input_file)
    os.rename(dump_file, input_file)
    range_file = os.path.join(os.path.dirname(input_file), 'range+' + os.path.basename(input_file))
    with open(range_file, 'wb') as fw:
        pickle.dump(class_value_range, fw)
        pickle.dump(class_act_range, fw)
