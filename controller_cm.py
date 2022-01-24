# -*-coding:utf8-*-

import torch
import numpy as np
import random
import copy
import parameter_tree_cm
import clustering_cm
import os
import pickle
import utils
import gc


class ReinforcementController(object):
    """
    used for maintain a set of parameters, and apply kernel operation
    (drop, merge, create)
    it also maintains a set of block-trees which is the structured module
    module training and evaluation is performed in this class
    """
    # part 1: initialization
    def __init__(self,
                 tree_volume,
                 learning_rate,
                 kernel_dim,
                 dim_shift,
                 max_layers,
                 input_dim,
                 dim_span,
                 num_class,
                 local_path,
                 channel=3,
                 threads=4,
                 use_cuda=False,
                 max_train_steps=10000,
                 early_stop=10,
                 evaluation_steps=1000,
                 add_rate=0.1,
                 cluster_rate=0.05,
                 dump_rate=0.5,
                 model_path=None,
                 param_path=None,
                 **kwargs):
        super(ReinforcementController, self).__init__(**kwargs)
        # system parameter
        self.local_path = local_path
        self.threads = threads
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        self.model_path = model_path
        self.param_path = param_path

        # structure parameters
        self.tree_volume = tree_volume
        self.kernel_dim = kernel_dim
        self.dim_shift = dim_shift
        self.dim_span = dim_span
        self.max_layers = max_layers
        self.input_dim = input_dim
        self.num_class = num_class
        self.channel = channel

        # cuda support
        self.use_cuda = use_cuda

        # model parameter
        self.block_dic = {}
        self.block_id = 0
        self.sequence_dic = {}  # a dict of sequence trees
        self.seq_id = 0

        # tuning additional parameter
        self.additional_bias = None
        self.train_additional_bias = None

        # initialization
        if self.model_path:
            self.load(model_path=self.model_path, param_path=None)
            for sid in self.sequence_dic.keys():
                if not os.path.exists(self.sequence_dic[sid].local_path):
                    os.makedirs(self.sequence_dic[sid].local_path)
        else:
            self.init_block_trees()

        # train parameters
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.early_stop = early_stop
        self.evaluation_steps = evaluation_steps

        # reinforcement control parameters
        self.add_rate = add_rate  # rate of add branch
        self.cluster_rate = cluster_rate
        self.dump_rate = dump_rate
        self.changed_branch = 0  # record number of changed branches

        # loss tolerance for drop branches during training, based on random choose target
        self.loss_tolerance = -np.log(1.0 / self.num_class) * 100
        self.branch_contribution = {}  # used to store branch contribution
        self.class_acc = {}
        self.class_prc = {}

        # support for refine branch processing
        # used for tag matched branches
        # key: reference sequence_id-branch_id-block_id-label
        # value: a set {target sequence_id-branch_id-block_id-label}
        self.matched_pairs = {}
        self.used_pairs = {}
        # map ref-key+tgt-key to match-distance
        self.dist_pairs = {}
        # key: ref_sid-ref_node, value: set(tgt_sid,-tgt_node)
        self.node_pairs = {}
        self.retrieval_files = set()

        # for data matching
        self.data_matched_pairs = {}
        self.data_used_pairs = {}
        self.data_dist_pairs = {}
        self.data_node_pairs = {}  # data node: channel_start1_start2
        data_match_path = os.path.join(self.local_path, 'data_match')
        if not os.path.exists(data_match_path):
            os.makedirs(data_match_path)
        self.range2sid = {}
        self.sample2count = {}

        # load from file
        if self.param_path:
            self.load(model_path=None, param_path=self.param_path)

    def add_blocks(self, block_num):
        """
        initialize block list, new blocks may be added latter
        :param block_num:
        :return:
        """
        start_id = self.block_id
        for i in range(block_num):
            self.block_dic[self.block_id] = (parameter_tree_cm.ParameterBlock(
                kernel_dim=self.kernel_dim
            ))
            self.block_id += 1
        end_id = self.block_id - 1
        return start_id, end_id

    def init_block_trees(self):
        """
        prepare initial model-tree structures
        each structure is a set of chains to link different blocks together
        :return:
        """
        dim_start1 = 0
        dim_start2 = 0
        while True:
            layers = self.max_layers
            dim_range = [dim_start1, dim_start1 + self.dim_span, dim_start2, dim_start2 + self.dim_span]
            for i in range(self.channel):
                self.sequence_dic[self.seq_id] = parameter_tree_cm.ParameterTree(
                    seq_id=self.seq_id,
                    num_class=self.num_class,
                    kernel_dim=self.kernel_dim,
                    local_path=os.path.join(self.local_path, 'sequences', str(self.seq_id)),
                    dim_range=dim_range,
                    channel=i
                )  # use prototype to store sequences
                start_block_id, end_block_id = self.add_blocks(block_num=layers)
                block_sequence = list(range(start_block_id, end_block_id + 1))
                self.sequence_dic[self.seq_id].append_sequence(new_seq=block_sequence, layer=0, stem=-1, detach=False)
                self.seq_id += 1
            if self.seq_id == self.tree_volume:
                break
            dim_start2 += self.dim_shift
            if (dim_start2 + self.dim_span) > self.input_dim[1]:
                dim_start2 = 0
                dim_start1 += self.dim_shift
                if (dim_start1 + self.dim_span) > self.input_dim[0]:
                    dim_start1 = 0

    def add_random_branches(self, branch_num):
        """
        add random branches
        :param branch_num:
        :return:
        """
        dim_ranges = set()
        for i in range(branch_num):
            channel = 0
            while True:
                dim_start1 = random.choice(list(range(self.input_dim[0] - self.dim_span)))
                dim_start2 = random.choice(list(range(self.input_dim[1] - self.dim_span)))
                if dim_start1 % self.dim_shift == 0 and dim_start2 % self.dim_shift == 0:
                    continue
                channel = random.choice(list(range(self.channel)))
                token = str(dim_start1) + '-' + str(dim_start2) + '-' + str(channel)
                if token not in dim_ranges:
                    dim_ranges.add(token)
                    break
            dim_range = [dim_start1, dim_start1 + self.dim_span, dim_start2, dim_start2 + self.dim_span]
            self.sequence_dic[self.seq_id] = parameter_tree_cm.ParameterTree(
                seq_id=self.seq_id,
                num_class=self.num_class,
                kernel_dim=self.kernel_dim,
                local_path=os.path.join(self.local_path, 'sequences', str(self.seq_id)),
                dim_range=dim_range,
                channel=channel
            )
            start_block_id, end_block_id = self.add_blocks(block_num=self.max_layers)
            block_sequence = list(range(start_block_id, end_block_id + 1))
            self.sequence_dic[self.seq_id].append_sequence(new_seq=block_sequence, layer=0, stem=-1, detach=False)
            self.seq_id += 1

    # part 2: forward computation
    def forward(self,
                inputs,
                seq_ids,
                mode='infer',
                joint=True,
                norm='relu',
                set_bound=False):
        """
        forward propagation of sequence trees
        :param inputs:
        :param mode:
        :param seq_ids:
        :param joint:
        :param norm:
        :param set_bound
        :return: class_output: classification output
        """
        # do branch forward propagation
        total_output_dict = {}
        if seq_ids:
            for sid in seq_ids.keys():
                branch_output_dict = self.sequence_dic[sid].forward(
                    block_dic=self.block_dic,
                    inputs=inputs, mode=mode,
                    branch_ids=seq_ids[sid],
                    norm=norm, set_bound=set_bound
                )
                total_output_dict.update(branch_output_dict)
        else:
            for sid in self.sequence_dic.keys():
                branch_output_dict = self.sequence_dic[sid].forward(
                    block_dic=self.block_dic,
                    inputs=inputs, mode=mode,
                    branch_ids=None,
                    norm=norm, set_bound=set_bound
                )
                total_output_dict.update(branch_output_dict)

        if joint:
            class_output = None
            for ti in total_output_dict.keys():
                if class_output is None:
                    class_output = total_output_dict[ti]
                else:
                    class_output += total_output_dict[ti]
            return class_output
        else:
            return total_output_dict

    # part 3: for training
    def train_sequences(self,
                        data_loader,
                        data_loader_dev,
                        tiny_dev,
                        seq_ids,
                        train_steps,
                        learning_rate,
                        eval_batch_size,
                        include_branch=True,
                        joint_training=True,
                        norm='relu'):
        """
        train sequences, for all sequences or for a subset
        :param data_loader: data loader
        :param data_loader_dev: evaluation data loader
        :param seq_ids: sequence to train, a dic, each element is the list of branch ids
        :param include_branch: whether train branch parameters
        :param train_steps: train steps for all training and additive training
        :param learning_rate: learning rate
        :param eval_batch_size:
        :param tiny_dev: tiny dev set for early stop
        :param joint_training: whether joint train all branches
        :param norm: activation function to apply to classifier of each branch
        :return:
        """
        # training
        loss_fn = torch.nn.CrossEntropyLoss()
        parameter_list = []
        if not seq_ids:
            for sid in self.sequence_dic.keys():
                seq_params = self.sequence_dic[sid].prepare_training(
                    branch_ids=None,
                    block_dic=self.block_dic,
                    include_branch=include_branch,
                    trainable=True,
                    use_cuda=self.use_cuda
                )
                parameter_list += seq_params
        else:
            for sid in seq_ids.keys():
                seq_params = self.sequence_dic[sid].prepare_training(
                    branch_ids=seq_ids[sid],
                    block_dic=self.block_dic,
                    include_branch=include_branch,
                    trainable=True,
                    use_cuda=self.use_cuda
                )
                parameter_list += seq_params
        print('\tnumber of training parameters:', len(parameter_list))
        opt = torch.optim.Adam(params=parameter_list, lr=learning_rate)
        non_best = 0
        step = 0
        best_train_loss = None
        best_loss_dic = {}  # map token to loss
        for data in data_loader:
            feature, label = data
            if self.use_cuda:
                feature = feature.cuda()
                label = label.cuda()
            opt.zero_grad()
            class_output = self.forward(
                inputs=feature,
                mode='train',
                seq_ids=seq_ids,
                joint=joint_training,
                norm=norm
            )
            if joint_training:
                loss = loss_fn(class_output, label)
            else:
                loss = None
                for ki in class_output.keys():
                    if loss is None:
                        loss = loss_fn(class_output[ki], label)
                    else:
                        loss = loss + loss_fn(class_output[ki], label)
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print('\tloss is NAN or INF, stop training')
                break
            loss.backward()
            opt.step()
            if (step * 10) % self.evaluation_steps == 0:  # for more detailed information
                print('\tLOG train: train loss at step', step, 'is', loss.item())
            if (step % self.evaluation_steps == 0) or (step + 1 == train_steps):
                if joint_training:
                    best_train_loss, non_best = self.eval_branch_joint(
                        step=step,
                        seq_ids=seq_ids,
                        tiny_dev=tiny_dev,
                        loss_fn=loss_fn,
                        best_train_loss=best_train_loss,
                        non_best=non_best,
                        norm=norm,
                        eval_batch_size=eval_batch_size
                    )
                else:
                    best_train_loss, non_best, best_loss_dic = self.eval_branch_separate(
                        step=step,
                        seq_ids=seq_ids,
                        tiny_dev=tiny_dev,
                        loss_fn=loss_fn,
                        best_train_loss=best_train_loss,
                        non_best=non_best,
                        best_loss_dic=best_loss_dic,
                        norm=norm,
                        eval_batch_size=eval_batch_size
                    )
                if non_best > self.early_stop:
                    break
            step += 1
            if step == train_steps:
                print('\tLOG train: train loss at last step', step, 'is', loss.item())
                break
        # load checkpoint and clear cuda
        self.clear_cuda()
        if seq_ids:
            for sid in seq_ids.keys():
                self.sequence_dic[sid].clear_cuda()
        else:
            for sid in self.sequence_dic.keys():
                self.sequence_dic[sid].clear_cuda()
        self.load_checkpoint(seq_ids=seq_ids)
        for bli in self.block_dic.keys():
            self.block_dic[bli].reset_train_state()
        # do evaluation
        best_loss, false_rate = self.branch_class_analysis(
            data_loader_dev=data_loader_dev,
            seq_ids=seq_ids,
            norm=norm
        )
        return best_loss, false_rate

    def eval_branch_joint(self,
                          step,
                          seq_ids,
                          tiny_dev,
                          loss_fn,
                          best_train_loss,
                          non_best,
                          norm,
                          eval_batch_size):
        """
        do evaluation on branches
        :return:
        """
        # refresh data loader
        tiny_dev.dataset.refresh_files()
        dev_samples = 0
        total_loss = 0
        with torch.no_grad():
            for feat_dev, lab_dev in tiny_dev:
                if self.use_cuda:
                    feat_dev = feat_dev.cuda()
                    lab_dev = lab_dev.cuda()
                # no need to compute reward here
                dev_class_output = self.forward(
                    inputs=feat_dev,
                    mode='eval',
                    seq_ids=seq_ids,
                    norm=norm
                )
                for i in range(eval_batch_size):
                    label_dev = lab_dev[i:i + 1]
                    forward_out = dev_class_output[i:i + 1, :]
                    fwd_loss = np.squeeze(loss_fn(forward_out, label_dev).detach().cpu().numpy())
                    total_loss += fwd_loss
                    dev_samples += 1
        avg_loss = total_loss / dev_samples
        print('\tLOG eval: train loss at step', step, 'is', avg_loss)
        if best_train_loss is None or avg_loss < best_train_loss:
            best_train_loss = avg_loss
            non_best = 0
            # save parameters to checkpoint
            print('\tLOG save checkpoint at step:', step)
            save_file = os.path.join(
                self.local_path, 'checkpoint_' + str(step) + '.pkl')
            save_block_dic = {}
            total_save_blocks = set()
            if seq_ids:
                for sid in seq_ids.keys():
                    self.sequence_dic[sid].save_classify_layer(
                        branch_ids=seq_ids[sid], step=step)
                    for bid in seq_ids[sid]:
                        blocks = self.sequence_dic[sid].branch_dic[bid]['sequence']
                        total_save_blocks = total_save_blocks.union(set(blocks))
            else:
                for sid in self.sequence_dic.keys():
                    self.sequence_dic[sid].save_classify_layer(
                        branch_ids=None, step=step)
                    for bid in self.sequence_dic[sid].branch_dic.keys():
                        blocks = self.sequence_dic[sid].branch_dic[bid]['sequence']
                        total_save_blocks = total_save_blocks.union(set(blocks))
            for bli in total_save_blocks:
                if self.block_dic[bli].excluded:
                    continue  # add support for excluded blocks
                save_block = copy.deepcopy(self.block_dic[bli])
                save_block.cpu(require_grad=False)
                save_block_dic[bli] = save_block
            with open(save_file, 'wb') as fw:
                pickle.dump(save_block_dic, fw)
        else:
            non_best += 1
        return best_train_loss, non_best

    def eval_branch_separate(self,
                             step,
                             seq_ids,
                             tiny_dev,
                             loss_fn,
                             best_train_loss,
                             best_loss_dic,
                             non_best,
                             norm,
                             eval_batch_size):
        """
        do evaluation on branches with individual loss
        in this case, just used in matched branches, there will not be saving blocks and drop branches
        :return:
        """
        # refresh data loader
        tiny_dev.dataset.refresh_files()
        dev_samples = 0
        cur_loss_dic = {}
        with torch.no_grad():
            for feat_dev, lab_dev in tiny_dev:
                if self.use_cuda:
                    feat_dev = feat_dev.cuda()
                    lab_dev = lab_dev.cuda()
                # no need to compute reward here
                dev_class_output = self.forward(
                    inputs=feat_dev,
                    mode='eval',
                    seq_ids=seq_ids,
                    joint=False,
                    norm=norm
                )
                for i in range(eval_batch_size):
                    for tk in dev_class_output.keys():
                        fwd_loss = np.squeeze(
                            loss_fn(dev_class_output[tk][i:i + 1], lab_dev[i:i + 1]).clone().detach().cpu().numpy())
                        if tk not in cur_loss_dic:
                            cur_loss_dic[tk] = fwd_loss
                        else:
                            cur_loss_dic[tk] += fwd_loss
                    dev_samples += 1
        save_seq_ids = {}
        avg_loss = 0
        for tk in cur_loss_dic.keys():
            cur_loss_dic[tk] /= dev_samples
            if tk not in best_loss_dic or best_loss_dic[tk] > cur_loss_dic[tk]:
                # choose better branches
                sid, bid = tk.split('_')
                if int(sid) not in save_seq_ids:
                    save_seq_ids[int(sid)] = [int(bid)]
                else:
                    save_seq_ids[int(sid)].append(int(bid))
                best_loss_dic[tk] = cur_loss_dic[tk]
            avg_loss += best_loss_dic[tk]
        print('\tLOG eval: train loss at step', step, 'is', avg_loss)
        if best_train_loss is None or avg_loss < best_train_loss:
            best_train_loss = avg_loss
            non_best = 0
            # save classifier
            if len(save_seq_ids) > 0:
                print('\tLOG save checkpoint at step:', step)
            save_blocks = set()
            for sid in save_seq_ids.keys():
                print('\tsave branch parameter:', sid, save_seq_ids[sid])
                self.sequence_dic[sid].save_classify_layer(branch_ids=save_seq_ids[sid], step=step)
                for bid in save_seq_ids[sid]:
                    sequence = self.sequence_dic[sid].branch_dic[bid]['sequence']
                    for bli in sequence:
                        if self.block_dic[bli].excluded:
                            continue
                        save_blocks.add(bli)
            save_file = os.path.join(
                self.local_path, 'checkpoint_' + str(step) + '.pkl')
            save_block_dic = {}
            for bli in save_blocks:
                save_block = copy.deepcopy(self.block_dic[bli])
                save_block.cpu(require_grad=False)
                save_block_dic[bli] = save_block
            with open(save_file, 'wb') as fw:
                pickle.dump(save_block_dic, fw)
        else:
            non_best += 1
        return best_train_loss, non_best, best_loss_dic

    def load_checkpoint(self, seq_ids):
        """
        load checkpoint of kernels, biases and bn-layers
        :return:
        """
        loaded_blocks = set()
        checkpoints = [fi for fi in os.listdir(self.local_path) if fi.startswith('checkpoint')]
        steps = []
        for ci in checkpoints:
            step = int(ci.split('.')[0].split('_')[1])
            steps.append(step)
        sorted_steps = sorted(steps, reverse=True)
        # load kernels or attentions
        for si in sorted_steps:
            save_file = os.path.join(
                self.local_path, 'checkpoint_' + str(si) + '.pkl')
            with open(save_file, 'rb') as fr:
                saved_block_dic = pickle.load(fr)
                for bli in saved_block_dic.keys():
                    if bli in loaded_blocks:
                        continue
                    self.block_dic[bli] = saved_block_dic[bli]
                    loaded_blocks.add(bli)
            os.remove(save_file)
        # load classify layers
        if seq_ids:
            for sid in seq_ids.keys():
                self.sequence_dic[sid].load_classify_layers()
        else:
            for sid in self.sequence_dic.keys():
                self.sequence_dic[sid].load_classify_layers()

    def clear_cuda(self):
        """
        clear cuda
        :return:
        """
        for bli in self.block_dic.keys():
            if self.block_dic[bli].is_cuda:
                self.block_dic[bli].cpu(require_grad=False)

    def branch_class_analysis(self, data_loader_dev, seq_ids, norm):
        """
        analyze recall and precision of classes for each branch
        just branches without class mask
        :param data_loader_dev:
        :param seq_ids:
        :param norm:
        :return:
        """
        data_loader_dev.dataset.refresh_files()
        if not seq_ids:
            seq_ids = {}
            for sid in self.sequence_dic.keys():
                seq_ids[sid] = list(self.sequence_dic[sid].branch_dic.keys())
        sample_count = 0
        class_dic = {}
        class_pred = {}
        for i in range(self.num_class):
            class_dic[i] = 0
            class_pred[i] = 0
        class_right = {}
        total_loss = None
        right_count = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        # do forward for each branch
        for dev_feat, dev_lab in data_loader_dev:
            with torch.no_grad():
                total_class_out = None
                # do forward propagation
                total_output_dict = {}
                for sid in seq_ids.keys():
                    seq_out_dict = self.sequence_dic[sid].forward(
                        inputs=dev_feat,
                        block_dic=self.block_dic,
                        mode='infer',
                        branch_ids=seq_ids[sid],
                        norm=norm
                    )
                    total_output_dict.update(seq_out_dict)
                true_lab = np.squeeze(dev_lab.numpy().astype(np.int32))
                class_dic[int(true_lab)] += 1
                for ti in total_output_dict.keys():
                    if total_class_out is None:
                        total_class_out = total_output_dict[ti]
                    else:
                        total_class_out = total_class_out + total_output_dict[ti]
                total_pred_np = np.squeeze(total_class_out.detach().numpy())
                total_pred_lab = np.argmax(total_pred_np)
                class_pred[int(total_pred_lab)] += 1
                max_value = np.max(total_pred_np)
                min_value = np.min(total_pred_np)
                for ti in total_output_dict.keys():
                    if ti not in self.branch_contribution:
                        self.branch_contribution[ti] = 0
                    class_out = np.squeeze(total_output_dict[ti].clone().detach().numpy())
                    contribution = 0
                    for cli in range(self.num_class):
                        if cli == int(true_lab):
                            contribution += (class_out[cli] - min_value) / (max_value - min_value)
                        else:
                            contribution -= (class_out[cli] - min_value) / (max_value - min_value)
                    self.branch_contribution[ti] += contribution
                if total_pred_lab == true_lab:
                    right_count += 1
                    if int(true_lab) not in class_right:
                        class_right[int(true_lab)] = 1
                    else:
                        class_right[int(true_lab)] += 1
                loss = np.squeeze(loss_fn(total_class_out, dev_lab).detach().numpy())
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
                sample_count += 1
        best_loss = total_loss / sample_count
        false_rate = 1 - (right_count / sample_count)
        for ti in self.branch_contribution.keys():
            self.branch_contribution[ti] /= sample_count
        for cli in class_dic.keys():
            print('\tclass:', cli, 'recall:', class_right[cli] / class_dic[cli])
            self.class_acc[cli] = class_right[cli] / class_dic[cli]
            print('\tclass:', cli, 'precision:', class_right[cli] / class_pred[cli])
            self.class_prc[cli] = class_right[cli] / class_pred[cli]
        return best_loss, false_rate

    # part 5: dump and cluster
    def get_input_range(self, data_loader_dev):
        """
        get input range of blocks
        :param data_loader_dev:
        :return:
        """
        data_loader_dev.dataset.refresh_files()
        for feat, lab in data_loader_dev:
            self.forward(inputs=feat, seq_ids=None, set_bound=True)

    def dump_and_cluster(self, seq_ids=None):
        """
        random dump points of branches
        :return:
        """
        data_loader_dev = utils.build_data_loader_uniform(kernel_dim=self.kernel_dim, batch_size=1)
        # randomly select branches to dump
        total_branches = []
        if not seq_ids:
            for sid in self.sequence_dic.keys():
                cluster_dir = os.path.join(self.local_path, 'sequences', str(sid), 'clusters')
                if os.path.exists(cluster_dir):
                    clustered_files = set([fi.split('.')[0] for fi in os.listdir(cluster_dir)])
                else:
                    clustered_files = set()
                for bid in self.sequence_dic[sid].branch_dic.keys():
                    flg_add = False
                    for bli in self.sequence_dic[sid].branch_dic[bid]['sequence']:
                        bl_tk = str(bid) + '_' + str(bli)
                        if bl_tk not in clustered_files:
                            flg_add = True
                            break
                    if flg_add:
                        token = str(sid) + '_' + str(bid)
                        total_branches.append(token)
        else:
            for sid in seq_ids.keys():
                cluster_dir = os.path.join(self.local_path, 'sequences', str(sid), 'clusters')
                if os.path.exists(cluster_dir):
                    clustered_files = set([fi.split('.')[0] for fi in os.listdir(cluster_dir)])
                else:
                    clustered_files = set()
                for bid in seq_ids[sid]:
                    flg_add = False
                    for bli in self.sequence_dic[sid].branch_dic[bid]['sequence']:
                        bl_tk = str(bid) + '_' + str(bli)
                        if bl_tk not in clustered_files:
                            flg_add = True
                            break
                    if flg_add:
                        token = str(sid) + '_' + str(bid)
                        total_branches.append(token)
        to_dump = int(self.tree_volume * self.dump_rate)
        if to_dump < len(total_branches):
            dump_branches = random.sample(total_branches, to_dump)
        else:
            dump_branches = total_branches
        dump_sequences = {}
        for ti in dump_branches:
            sid, bid = ti.split('_')
            sid = int(sid)
            bid = int(bid)
            if sid not in dump_sequences:
                dump_sequences[sid] = [bid]
            else:
                dump_sequences[sid].append(bid)
        # dump points
        total_to_dump = 0
        new_dump_files = []
        for sid in dump_sequences.keys():
            out_files = self.sequence_dic[sid].prepare_dumps(
                branch_ids=dump_sequences[sid]
            )  # prepare dump
            new_dump_files += out_files
            total_to_dump += len(out_files)
        if total_to_dump == 0:  # no need to do forward
            return
        with torch.no_grad():
            sample_id = 0
            for dev_feat, dev_lab in data_loader_dev:
                for sid in dump_sequences.keys():
                    self.sequence_dic[sid].target_forward_dump(
                        inputs=dev_feat,
                        block_dic=self.block_dic,
                        sample_id=sample_id,
                        norm='none'
                    )
                sample_id += 1
                if sample_id % 1000 == 0:
                    print('\t dump sample id:', sample_id)
                if sample_id == 10000:
                    break
        for sid in dump_sequences.keys():
            self.sequence_dic[sid].delete_dumps()  # close dumps
        for fi in new_dump_files:
            parameter_tree_cm.select_best_samples(input_file=fi, min_samples=500, sample_rate=0.1)
        # check out files
        checked_files = []
        for oi in new_dump_files:
            with open(oi, 'rb') as fr:
                sp_count = 0
                while True:
                    try:
                        sp = pickle.load(fr)
                        if not isinstance(sp, list):
                            print('\twrong data format')
                            break
                        sp_count += 1
                    except EOFError:
                        break
                if sp_count > 0:
                    checked_files.append(oi)
                else:
                    print('\tthere is no dump in file:', oi)
        # do cluster
        out_dirs = []
        for ci in checked_files:
            out_dir = os.path.join('/'.join(ci.split('/')[:-2]), 'clusters')
            out_dirs.append(out_dir)
        cluster_files = clustering_cm.parallel_cluster(
            input_files=checked_files,
            threads=self.threads,
            out_dirs=out_dirs,
            ref=False
        )
        sid2clusters = {}
        for ci in cluster_files:
            sid = int(ci.split('/')[-3])
            if sid not in sid2clusters:
                sid2clusters[sid] = [ci]
            else:
                sid2clusters[sid].append(ci)
        # update retrieval file
        print('\tstart to update retrieval file')
        for sid in self.sequence_dic.keys():
            if sid not in sid2clusters:
                continue
            retrieval_file = os.path.join(self.sequence_dic[sid].local_path, 'retrieval.pkl')
            self.retrieval_files.add(retrieval_file)
            clustering_cm.update_clusters(retrieval_file=retrieval_file, input_files=sid2clusters[sid])
        print('\tclustering done')
        return cluster_files

    def ref_cluster_dump(self, data_loader_dev, seq_ids=None):
        """
        dump reference clusters
        :param data_loader_dev:
        :param seq_ids:
        :return:
        """
        data_loader_dev.dataset.refresh_files()
        all_branches_tokens = []
        if not seq_ids:
            for sid in self.sequence_dic.keys():
                cluster_dir = os.path.join(self.local_path, 'sequences', str(sid), 'ref_clusters')
                if os.path.exists(cluster_dir):
                    cluster_files = set([fi.split('.')[0] for fi in os.listdir(cluster_dir)])
                else:
                    cluster_files = set()
                for bid in self.sequence_dic[sid].branch_dic.keys():
                    flg_add = False
                    for bli in self.sequence_dic[sid].branch_dic[bid]['sequence']:
                        bl_tk = str(bid) + '_' + str(bli)
                        if bl_tk not in cluster_files:
                            flg_add = True
                            break
                    if flg_add:
                        all_branches_tokens.append(str(sid) + '_' + str(bid))
        else:
            for sid in seq_ids.keys():
                cluster_dir = os.path.join(self.local_path, 'sequences', str(sid), 'ref_clusters')
                if os.path.exists(cluster_dir):
                    cluster_files = set([fi.split('.')[0] for fi in os.listdir(cluster_dir)])
                else:
                    cluster_files = set()
                for bid in seq_ids[sid]:
                    flg_add = False
                    for bli in self.sequence_dic[sid].branch_dic[bid]['sequence']:
                        bl_tk = str(bid) + '_' + str(bli)
                        if bl_tk not in cluster_files:
                            flg_add = True
                            break
                    if flg_add:
                        all_branches_tokens.append(str(sid) + '_' + str(bid))
        to_match = int(self.tree_volume * self.dump_rate)
        if to_match < len(all_branches_tokens):
            ref_branches = random.sample(all_branches_tokens, to_match)
        else:
            ref_branches = all_branches_tokens
        # compute ref-cluster and dump in temp path
        sid_set = set()
        dump_seq_ids = {}
        for ri in ref_branches:
            sid, bid = ri.split('_')
            sid = int(sid)
            sid_set.add(sid)
            bid = int(bid)
            if sid not in dump_seq_ids:
                dump_seq_ids[sid] = [bid]
            else:
                dump_seq_ids[sid].append(bid)
        ref_dumps = []
        for sid in dump_seq_ids.keys():
            ref_dump = self.sequence_dic[sid].prepare_dumps(
                branch_ids=dump_seq_ids[sid],
                temp=True
            )
            ref_dumps += ref_dump
        with torch.no_grad():  # do forward dump
            sample_id = 0
            for dev_feat, dev_lab in data_loader_dev:
                for sid in dump_seq_ids.keys():
                    self.sequence_dic[sid].ref_forward_dump(
                        inputs=dev_feat,
                        block_dic=self.block_dic,
                        sample_id=sample_id,
                        label=dev_lab
                    )
                sample_id += 1
        for sid in dump_seq_ids.keys():
            self.sequence_dic[sid].delete_dumps()
        # check out files
        checked_files = []
        for oi in ref_dumps:
            with open(oi, 'rb') as fr:
                sp_count = 0
                while True:
                    try:
                        sp = pickle.load(fr)
                        if not isinstance(sp, list):
                            print('\twrong data format')
                            break
                        sp_count += 1
                    except EOFError:
                        break
                if sp_count > 0:
                    checked_files.append(oi)
                else:
                    print('\tthere is no dump in file:', oi)
        return checked_files

    # part 6: branch matching
    def match_branches(self, reference_files, to_match, refine_class):
        """
        do match on reference node and target node in pair
        :param reference_files: all reference clusters (file path)
        :param to_match: number of reference or target clusters to match (file path)
        :param refine_class: reference class to match
        :return:
        """
        count = 0
        for ri in reference_files:
            ref_sid = ri.split('/')[-3]
            ref_node = ri.split('/')[-1].split('.')[0]
            ref_cluster_file = os.path.join(
                self.sequence_dic[int(ref_sid)].local_path, 'ref_clusters', ref_node + '.pkl')
            ref_clusters = []
            with open(ref_cluster_file, 'rb') as fr:
                while True:
                    try:
                        dump_cluster = pickle.load(fr)
                        ref_cluster, ref_stat, label = dump_cluster
                        if int(label) == refine_class:
                            ref_clusters.append(ref_cluster)
                    except EOFError:
                        break
            # retrieve clusters
            if len(ref_clusters) == 0:
                continue
            print('\tstart retieve reference cluster:', ref_sid, ref_node, len(ref_clusters))
            retrieved_keys = clustering_cm.parallel_retrieve(
                ref_clusters=ref_clusters,
                retrieval_files=self.retrieval_files,
                ref_sid=ref_sid,
                ref_node=ref_node,
                beam=2,
                matched_pairs=self.matched_pairs,
                to_match_label=refine_class,
                threads=self.threads
            )
            tgt_to_match = set()
            ref_node_key = ref_sid + '-' + ref_node
            if ref_node_key not in self.node_pairs:
                self.node_pairs[ref_node_key] = set()
            for rki in retrieved_keys:
                target_sid, target_node, target_label = rki.split('+')
                tgt_node_key = target_sid + '-' + target_node
                if tgt_node_key in self.node_pairs[ref_node_key]:
                    continue
                else:
                    self.node_pairs[ref_node_key].add(tgt_node_key)
                target_file = os.path.join(
                    self.sequence_dic[int(target_sid)].local_path, 'clusters', target_node + '.pkl')
                tgt_to_match.add(target_file)
            tgt_to_match = list(tgt_to_match)
            print('\tstart matching target cluster:', len(tgt_to_match))
            matched_dict, dist_dict = clustering_cm.parallel_class_match(
                ref_file=ri,
                target_files=tgt_to_match,
                threads=self.threads,
                num_class=self.num_class
            )
            self.dist_pairs.update(dist_dict)
            # update matched pairs
            for rk in matched_dict.keys():
                if rk not in self.matched_pairs:
                    self.matched_pairs[rk] = set()
                for tk in matched_dict[rk]:
                    self.matched_pairs[rk].add(tk)
            count += 1
            if count == to_match:
                break

    def match_branches_data(self,
                            data_files,
                            to_match,
                            target_clusters,
                            sub_sample=0):
        """
        do match on reference node and target node in pair
        :param data_files: all reference clusters (file path)
        :param to_match: number of reference or target clusters to match (file path)
        :param target_clusters: all target clusters
        :param sub_sample:
        :return:
        """
        count = 0
        match_weight = self.make_match_weight()
        for di in data_files:
            basename = os.path.basename(di).split('.')[0]
            # channel, width_start, height_start = basename.split('_')
            ref_sid = '_1'
            ref_node = basename
            tgt_to_match = set()
            ref_node_key = ref_sid + '-' + ref_node
            if ref_node_key not in self.data_node_pairs:
                self.data_node_pairs[ref_node_key] = set()
            for tci in target_clusters:
                target_sid = tci.split('/')[-3]
                target_node = os.path.basename(tci).split('.')[0]
                tgt_node_key = target_sid + '-' + target_node
                if tgt_node_key in self.data_node_pairs[ref_node_key]:
                    continue
                target_file = tci
                tgt_to_match.add(target_file)
            tgt_to_match = list(tgt_to_match)
            if len(tgt_to_match) == 0:
                continue
            if 0 < sub_sample < len(tgt_to_match):
                tgt_to_match = random.sample(tgt_to_match, sub_sample)
            # update data node pairs
            for tci in tgt_to_match:
                target_sid = tci.split('/')[-3]
                target_node = os.path.basename(tci).split('.')[0]
                tgt_node_key = target_sid + '-' + target_node
                self.data_node_pairs[ref_node_key].add(tgt_node_key)
            print('\tstart matching target cluster:', len(tgt_to_match))
            matched_dict, dist_dict = clustering_cm.parallel_class_match(
                ref_file=di,
                target_files=tgt_to_match,
                threads=self.threads,
                num_class=self.num_class,
                match_data=True,
                dump_path=os.path.join(self.local_path, 'data_match'),
                sample_rate=1.0,
                sample_weight=match_weight
            )
            self.data_dist_pairs.update(dist_dict)
            # update matched pairs
            for rk in matched_dict.keys():
                if rk not in self.data_matched_pairs:
                    self.data_matched_pairs[rk] = set()
                for tk in matched_dict[rk]:
                    self.data_matched_pairs[rk].add(tk)
            count += 1
            if count == to_match:
                break
        return count

    def use_branches(self, to_use, refine_class=None):
        """
        use matched branches for creating new branches
        :param to_use
        :param refine_class:
        :return:
        """
        use_count = 0
        seq_ids = {}
        # sort matched dist and select best ones
        unused_pairs = {}  # key: ref-key+tgt-key, value: distance
        for ref_key in self.matched_pairs:
            ref_label = int(ref_key.split('-')[-1])
            if refine_class is not None and ref_label != refine_class:
                continue
            for tgt_key in self.matched_pairs[ref_key]:
                if ref_key in self.used_pairs and tgt_key in self.used_pairs[ref_key]:
                    continue
                dist_key = ref_key + '+' + tgt_key
                unused_pairs[dist_key] = self.dist_pairs[dist_key]
        sorted_dists = sorted(unused_pairs.items(), key=lambda x: x[1])
        to_use = min(to_use, len(sorted_dists))
        if to_use == 0:
            print('\tto use count is 0')
            return seq_ids, use_count
        to_use_pairs = {}
        for i in range(to_use):
            ref_key, tgt_key = sorted_dists[i][0].split('+')
            print('\tmatch distance:', ref_key, tgt_key, sorted_dists[i][1])
            if ref_key not in to_use_pairs:
                to_use_pairs[ref_key] = set()
            to_use_pairs[ref_key].add(tgt_key)
        # start using matched branches
        for ref_key in to_use_pairs.keys():
            ref_sid, ref_node, ref_label = ref_key.split('-')
            ref_label = int(ref_label)
            ref_sid = int(ref_sid)
            ref_block = int(ref_node.split('_')[1])
            ref_bid = int(ref_node.split('_')[0])
            if ref_key not in self.used_pairs:
                self.used_pairs[ref_key] = set()
            for tk in to_use_pairs[ref_key]:
                self.used_pairs[ref_key].add(tk)
            flg_brk = False
            for tk in to_use_pairs[ref_key]:
                tgt_sid, tgt_node, tgt_label = tk.split('-')
                tgt_sid = int(tgt_sid)
                tgt_label = int(tgt_label)
                # load statistic
                base_name = ref_key + '+' + tk + '.pkl'
                stat_file = os.path.join(
                    self.sequence_dic[ref_sid].local_path,
                    'ref_clusters',
                    base_name
                )
                if not os.path.exists(stat_file):
                    print('\tno stat file:', stat_file)
                    continue
                with open(stat_file, 'rb') as fr:
                    ref_statistic = pickle.load(fr)
                    tgt_statistic = pickle.load(fr)
                    input_boundary = pickle.load(fr)
                print('\tmatched pairs:', ref_key, tk)
                print('\tremove file:', stat_file)
                os.remove(stat_file)
                # process transfer parameter
                ref_center, ref_dim_scales, ref_sort_dims = ref_statistic
                target_mean, target_dim_scales, target_sort_dims = tgt_statistic
                trans_dim_order = np.zeros(shape=[self.kernel_dim], dtype=np.int32)
                for i in range(self.kernel_dim):
                    target_order = target_sort_dims[i]
                    ref_order = ref_sort_dims[i]
                    trans_dim_order[ref_order] = target_order
                trans_dim_scale = np.zeros(shape=[self.kernel_dim], dtype=np.float32)
                for i in range(self.kernel_dim):
                    ref_dim = i
                    target_dim = trans_dim_order[i]
                    scale = (ref_dim_scales[ref_dim] + 1e-10) / (target_dim_scales[target_dim] + 1e-10)
                    if scale > 1e4:
                        scale = 1.0
                    trans_dim_scale[target_dim] = scale
                # use maximum for best data range
                trans_dim_shift = np.zeros(shape=[self.kernel_dim], dtype=np.float32)
                for i in range(self.kernel_dim):
                    ref_dim = i
                    target_dim = trans_dim_order[i]
                    trans_dim_shift[target_dim] += ref_center[ref_dim]
                # initialize new branches
                # get layer of ref-block
                _, ref_layer, ref_start_branch = self.sequence_dic[ref_sid].trace_blocks(
                    branch_id=ref_bid, target_block=ref_block)
                ref_layer += 1
                # get sequence of target branch
                target_bid = int(tgt_node.split('_')[0])
                target_block = int(tgt_node.split('_')[1])
                target_block_sequence, target_start_layer, target_start_branch = \
                    self.sequence_dic[tgt_sid].trace_blocks(
                        branch_id=target_bid, target_block=target_block)
                print('\ttarget block sequence length is:', len(target_block_sequence))
                match_ref_sids = [ref_sid]
                match_ref_start_branches = [ref_start_branch]
                match_ref_layers = [ref_layer]
                match_trans_dim_orders = [trans_dim_order]
                match_trans_dim_scales = [trans_dim_scale]
                match_trans_dim_shifts = [trans_dim_shift]
                to_match_labels = [ref_label]
                new_sids, new_bids = self.init_from_branch(
                    ref_sids=match_ref_sids,
                    ref_bids=match_ref_start_branches,
                    ref_start_layers=match_ref_layers,
                    append_blocks=target_block_sequence,
                    target_sid=tgt_sid,
                    target_bid=target_bid,
                    target_class=tgt_label,
                    trans_dim_orders=match_trans_dim_orders,
                    trans_dim_scales=match_trans_dim_scales,
                    trans_dim_shifts=match_trans_dim_shifts,
                    target_center=target_mean,
                    matched_labels=to_match_labels,
                    boundary=input_boundary
                )
                for ids, sid in enumerate(new_sids):
                    if sid not in seq_ids:
                        seq_ids[sid] = [new_bids[ids]]
                    else:
                        seq_ids[sid].append(new_bids[ids])
                use_count += 1
                if use_count == to_use:
                    flg_brk = True
                    break
            if flg_brk:
                break
        # return seq_ids and use_count
        return seq_ids, use_count

    def use_branches_data(self, to_use, refine_class=None):
        """
        use matched branches for creating new branches
        :param to_use
        :param refine_class:
        :return:
        """
        use_count = 0
        seq_ids = {}
        # sort matched dist and select best ones
        unused_pairs = {}  # key: ref-key+tgt-key, value: distance
        for ref_key in self.data_matched_pairs:
            ref_label = int(ref_key.split('-')[-1])
            if refine_class is not None and ref_label != refine_class:
                continue
            for tgt_key in self.data_matched_pairs[ref_key]:
                if ref_key in self.data_used_pairs and tgt_key in self.data_used_pairs[ref_key]:
                    continue
                dist_key = ref_key + '+' + tgt_key
                unused_pairs[dist_key] = self.data_dist_pairs[dist_key]
        sorted_dists = sorted(unused_pairs.items(), key=lambda x: x[1])
        to_use = min(to_use, len(sorted_dists))
        if to_use == 0:
            print('\tto use count is 0')
            return seq_ids, use_count
        to_use_pairs = {}
        for i in range(to_use):
            ref_key, tgt_key = sorted_dists[i][0].split('+')
            # print('\tmatch distance:', ref_key, tgt_key, sorted_dists[i][1])
            if ref_key not in to_use_pairs:
                to_use_pairs[ref_key] = set()
            to_use_pairs[ref_key].add(tgt_key)
        # start using matched branches
        for ref_key in to_use_pairs.keys():
            _, ref_node, ref_label = ref_key.split('-')
            ref_label = int(ref_label)
            if ref_key not in self.data_used_pairs:
                self.data_used_pairs[ref_key] = set()
            for tk in to_use_pairs[ref_key]:
                self.data_used_pairs[ref_key].add(tk)
            flg_brk = False
            for tk in to_use_pairs[ref_key]:
                tgt_sid, tgt_node, tgt_label = tk.split('-')
                tgt_sid = int(tgt_sid)
                tgt_label = int(tgt_label)
                # load statistic
                base_name = ref_key + '+' + tk + '.pkl'
                stat_file = os.path.join(
                    self.local_path,
                    'data_match',
                    ref_node,
                    base_name
                )
                if not os.path.exists(stat_file):
                    print('\tno stat file:', stat_file)
                    continue
                with open(stat_file, 'rb') as fr:
                    ref_statistic = pickle.load(fr)
                    tgt_statistic = pickle.load(fr)
                    input_boundary = pickle.load(fr)
                os.remove(stat_file)
                # process transfer parameter
                ref_center, ref_dim_scales, ref_sort_dims = ref_statistic
                target_mean, target_dim_scales, target_sort_dims = tgt_statistic
                trans_dim_order = np.zeros(shape=[self.kernel_dim], dtype=np.int32)
                for i in range(self.kernel_dim):
                    target_order = target_sort_dims[i]
                    ref_order = ref_sort_dims[i]
                    trans_dim_order[ref_order] = target_order
                trans_dim_scale = np.zeros(shape=[self.kernel_dim], dtype=np.float32)
                for i in range(self.kernel_dim):
                    ref_dim = i
                    target_dim = trans_dim_order[i]
                    scale = (ref_dim_scales[ref_dim] + 1e-10) / (target_dim_scales[target_dim] + 1e-10)
                    if scale > 1e4:
                        scale = 1.0
                    trans_dim_scale[target_dim] = scale
                # use maximum for best data range
                trans_dim_shift = np.zeros(shape=[self.kernel_dim], dtype=np.float32)
                for i in range(self.kernel_dim):
                    ref_dim = i
                    target_dim = trans_dim_order[i]
                    trans_dim_shift[target_dim] += ref_center[ref_dim]
                # initialize new branches
                # get sequence of target branch
                target_bid = int(tgt_node.split('_')[0])
                target_block = int(tgt_node.split('_')[1])
                target_block_sequence, target_start_layer, target_start_branch = \
                    self.sequence_dic[tgt_sid].trace_blocks(
                        branch_id=target_bid, target_block=target_block)
                new_sids, new_bids = self.init_from_branch_data(
                    ref_node=ref_node,
                    append_blocks=target_block_sequence,
                    target_sid=tgt_sid,
                    target_bid=target_bid,
                    target_class=tgt_label,
                    trans_dim_order=trans_dim_order,
                    trans_dim_scale=trans_dim_scale,
                    trans_dim_shift=trans_dim_shift,
                    target_center=target_mean,
                    matched_label=ref_label,
                    boundary=input_boundary
                )
                for ids, sid in enumerate(new_sids):
                    if sid not in seq_ids:
                        seq_ids[sid] = [new_bids[ids]]
                    else:
                        seq_ids[sid].append(new_bids[ids])
                use_count += 1
                if use_count == to_use:
                    flg_brk = True
                    break
            if flg_brk:
                break
        # return seq_ids and use_count
        return seq_ids, use_count

    def init_from_branch(self,
                         ref_sids,
                         ref_bids,
                         ref_start_layers,
                         append_blocks,
                         target_sid,
                         target_bid,
                         target_class,
                         trans_dim_orders,
                         trans_dim_scales,
                         trans_dim_shifts,
                         target_center,
                         matched_labels,
                         boundary):
        """
        initialize new branch from target branch and append for reference branch
        :param ref_sids:
        :param ref_bids:
        :param ref_start_layers:
        :param append_blocks:
        :param target_sid:
        :param target_bid:
        :param target_class:
        :param trans_dim_orders:
        :param trans_dim_scales:
        :param trans_dim_shifts:
        :param target_center:
        :param matched_labels:
        :param boundary:
        :return:
        """
        new_sids = []
        new_bids = []
        # add new kernels to kernel dic and initialize new block
        classifier_weight = copy.deepcopy(
            self.sequence_dic[target_sid].branch_dic[target_bid]['classify_layer']
                .weight.clone().detach().numpy()[target_class, :])
        classifier_weight = np.expand_dims(classifier_weight, axis=0)
        classifier_bias = copy.deepcopy(
            self.sequence_dic[target_sid].branch_dic[target_bid]['classify_layer']
                .bias.clone().detach().numpy()[target_class])
        for idx, ref_sid in enumerate(ref_sids):
            # do kernel and bias transformation
            block_sequence = []
            to_transform_kernels = self.block_dic[append_blocks[0]].linear.weight.clone().detach().numpy()
            to_transform_bias = self.block_dic[append_blocks[0]].linear.bias.clone().detach().numpy()
            transfer_result = parameter_tree_cm.parameter_transfer_numpy(
                kernel=to_transform_kernels,
                bias=to_transform_bias,
                trans_dim_order=trans_dim_orders[idx],
                trans_dim_scale=np.expand_dims(trans_dim_scales[idx], axis=0),
                trans_dim_shift=trans_dim_shifts[idx],
                target_center=target_center
            )
            if transfer_result is None:
                # continue to next added branch
                continue
            self.block_dic[self.block_id] = (parameter_tree_cm.ParameterBlock(
                kernel_dim=self.kernel_dim,
                weight=transfer_result[0],
                bias=transfer_result[1]
            ))
            self.block_dic[self.block_id].set_bound(
                inputs=torch.tensor(np.expand_dims(boundary[0], axis=0), dtype=torch.float32))
            self.block_dic[self.block_id].set_bound(
                inputs=torch.tensor(np.expand_dims(boundary[1], axis=0), dtype=torch.float32))
            block_sequence.append(self.block_id)
            self.block_id += 1
            for i in range(1, len(append_blocks)):
                block_id = append_blocks[i]
                self.block_dic[block_id].exclude()
                block_sequence.append(block_id)
                # for parameter sharing
                self.block_dic[block_id].add_reference()
            # add branch to parameter tree
            new_ref_branch = self.sequence_dic[ref_sid].append_sequence(
                new_seq=block_sequence,
                layer=ref_start_layers[idx],
                stem=ref_bids[idx],
                detach=True
            )
            print('\tnew_branch is:', ref_sid, new_ref_branch)
            # refine class
            self.sequence_dic[ref_sid].add_class_mask(
                branch_id=new_ref_branch,
                classes=[matched_labels[idx]],
                weight=classifier_weight,
                bias=classifier_bias
            )
            new_sids.append(ref_sid)
            new_bids.append(new_ref_branch)
        return new_sids, new_bids

    def init_from_branch_data(self,
                              ref_node,
                              append_blocks,
                              target_sid,
                              target_bid,
                              target_class,
                              trans_dim_order,
                              trans_dim_scale,
                              trans_dim_shift,
                              target_center,
                              matched_label,
                              boundary):
        """
        initialize new branch from target branch and append for reference branch
        :param ref_node:
        :param append_blocks:
        :param target_sid:
        :param target_bid:
        :param target_class:
        :param trans_dim_order:
        :param trans_dim_scale:
        :param trans_dim_shift:
        :param target_center:
        :param matched_label:
        :param boundary:
        :return:
        """
        new_sids = []
        new_bids = []
        # add new kernels to kernel dic and initialize new block
        classifier_weight = copy.deepcopy(
            self.sequence_dic[target_sid].branch_dic[target_bid]['classify_layer']
                .weight.clone().detach().numpy()[target_class, :])
        classifier_weight = np.expand_dims(classifier_weight, axis=0)
        classifier_bias = copy.deepcopy(
            self.sequence_dic[target_sid].branch_dic[target_bid]['classify_layer']
                .bias.clone().detach().numpy()[target_class])
        # do kernel and bias transformation
        block_sequence = []
        to_transform_kernels = self.block_dic[append_blocks[0]].linear.weight.clone().detach().numpy()
        to_transform_bias = self.block_dic[append_blocks[0]].linear.bias.clone().detach().numpy()
        transfer_result = parameter_tree_cm.parameter_transfer_numpy(
            kernel=to_transform_kernels,
            bias=to_transform_bias,
            trans_dim_order=trans_dim_order,
            trans_dim_scale=np.expand_dims(trans_dim_scale, axis=0),
            trans_dim_shift=trans_dim_shift,
            target_center=target_center
        )
        if transfer_result is None:
            # continue to next added branch
            return [], []
        self.block_dic[self.block_id] = (parameter_tree_cm.ParameterBlock(
            kernel_dim=self.kernel_dim,
            weight=transfer_result[0],
            bias=transfer_result[1]
        ))
        self.block_dic[self.block_id].set_bound(
            inputs=torch.tensor(np.expand_dims(boundary[0], axis=0), dtype=torch.float32))
        self.block_dic[self.block_id].set_bound(
            inputs=torch.tensor(np.expand_dims(boundary[1], axis=0), dtype=torch.float32))
        block_sequence.append(self.block_id)
        self.block_id += 1
        for i in range(1, len(append_blocks)):
            block_id = append_blocks[i]
            self.block_dic[block_id].exclude()
            block_sequence.append(block_id)
            # for parameter sharing
            self.block_dic[block_id].add_reference()
        # add new parameter tree
        channel, dim_start1, dim_start2 = ref_node.split('_')
        channel = int(channel)
        dim_start1 = int(dim_start1)
        dim_start2 = int(dim_start2)
        dim_range = [dim_start1, dim_start1 + self.dim_span, dim_start2, dim_start2 + self.dim_span]
        if ref_node not in self.range2sid:
            self.sequence_dic[self.seq_id] = parameter_tree_cm.ParameterTree(
                seq_id=self.seq_id,
                num_class=self.num_class,
                kernel_dim=self.kernel_dim,
                local_path=os.path.join(self.local_path, 'sequences', str(self.seq_id)),
                dim_range=dim_range,
                channel=channel
            )  # use prototype to store sequences
            ref_sid = self.seq_id
            self.range2sid[ref_node] = ref_sid
            self.seq_id += 1
        else:
            ref_sid = self.range2sid[ref_node]
        new_ref_branch = self.sequence_dic[ref_sid].append_sequence(
            new_seq=block_sequence,
            layer=0,
            stem=-1,
            detach=False
        )
        # refine class
        self.sequence_dic[ref_sid].add_class_mask(
            branch_id=new_ref_branch,
            classes=[matched_label],
            weight=classifier_weight,
            bias=classifier_bias
        )
        new_sids.append(ref_sid)
        new_bids.append(new_ref_branch)
        return new_sids, new_bids

    def search_branch_results(self,
                              out_dir,
                              data_loader_dev,
                              seq_ids,
                              matched_class,
                              min_precision,
                              min_recall,
                              batch_size,
                              tune_eval_path,
                              do_tuning=False):
        """
        :return:
        """
        # print('\tsearch seq ids:', seq_ids)
        if len(seq_ids) == 0:
            return seq_ids
        # make dump files
        out_path = os.path.join(self.local_path, out_dir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        dump_files = {}  # map token to file-object
        all_dump_files = set()
        search_branches = 0
        for sid in seq_ids.keys():
            self.sequence_dic[sid].set_graph(branch_ids=seq_ids[sid])
            for bid in seq_ids[sid]:
                # disable class-mask
                self.sequence_dic[sid].branch_dic[bid]['class_mask'].disable()
                token = str(sid) + '_' + str(bid)
                out_file = os.path.join(out_path, token + '.pkl')
                all_dump_files.add(out_file)
                fw = open(out_file, 'wb')
                dump_files[token] = fw
                search_branches += 1
        # forward computation
        data_loader_dev.dataset.refresh_files()
        target_count = 0
        with torch.no_grad():
            sample_count = 0
            for feat_dev, lab_dev in data_loader_dev:
                # no need to compute reward here
                total_out_dic = {}
                for sid in seq_ids.keys():
                    dev_class_output = self.sequence_dic[sid].forward(
                        block_dic=self.block_dic,
                        inputs=feat_dev,
                        branch_ids=seq_ids[sid],
                        mode='infer',
                        norm='none',
                        set_bound=False,
                        use_bound=True
                    )
                    total_out_dic.update(dev_class_output)
                for i in range(batch_size):
                    label = np.squeeze(lab_dev.numpy())[i].astype(np.int32)
                    if label == matched_class:
                        true_label = 1
                        target_count += 1
                    else:
                        true_label = 0
                    for tk in total_out_dic.keys():
                        class_out = np.squeeze(total_out_dic[tk][0].clone().detach().numpy())[i]
                        cmp_res = bool(np.squeeze(total_out_dic[tk][1].clone().detach().numpy())[i])
                        if not cmp_res:
                            if true_label == 1:
                                pickle.dump([class_out, 1, sample_count], dump_files[tk])
                            else:
                                pickle.dump([class_out, 0, sample_count], dump_files[tk])
                    sample_count += 1
            eval_basename = random.choice(os.listdir(tune_eval_path))
            eval_file = os.path.join(tune_eval_path, eval_basename)
            # make eval results
            data_loader_eval = utils.build_data_loader_from_file(
                data_path=eval_file,
                batch_size=batch_size,
                repeat=False
            )
            for feat_dev, lab_dev in data_loader_eval:
                # no need to compute reward here
                total_out_dic = {}
                for sid in seq_ids.keys():
                    dev_class_output = self.sequence_dic[sid].forward(
                        block_dic=self.block_dic,
                        inputs=feat_dev,
                        branch_ids=seq_ids[sid],
                        mode='infer',
                        norm='none',
                        set_bound=False,
                        use_bound=True
                    )
                    total_out_dic.update(dev_class_output)
                for i in range(batch_size):
                    label = np.squeeze(lab_dev.numpy())[i].astype(np.int32)
                    if label == matched_class:
                        true_label = 1
                    else:
                        true_label = 0
                    for tk in total_out_dic.keys():
                        class_out = np.squeeze(total_out_dic[tk][0].clone().detach().numpy())[i]
                        cmp_res = bool(np.squeeze(total_out_dic[tk][1].clone().detach().numpy())[i])
                        if not cmp_res:
                            if true_label == 1:
                                pickle.dump([class_out, 1, sample_count], dump_files[tk])
                            else:
                                pickle.dump([class_out, 0, sample_count], dump_files[tk])
                    sample_count += 1
            del data_loader_eval
        for tk in dump_files.keys():
            dump_files[tk].close()
        # make sample to weight
        sample2weight = self.make_sample2weight(matched_class=matched_class)
        # search best profit
        drop_seq_ids = {}
        drop_count = 0
        remained_seq_ids = {}
        for fi in all_dump_files:
            token = os.path.basename(fi).split('.')[0]
            search_res = utils.get_target_threshold_cmp_eval(
                input_file=fi,
                target_count=target_count,
                min_precision=min_precision,
                min_recall=min_recall,
                sample2weight=sample2weight,
                do_tuning=do_tuning
            )
            sid, bid = token.split('_')
            sid = int(sid)
            bid = int(bid)
            if search_res is None:
                drop_count += 1
                if sid in drop_seq_ids:
                    drop_seq_ids[sid].append(bid)
                else:
                    drop_seq_ids[sid] = [bid]
            else:
                recall, precision, profit, threshold, value_span, selected_target_samples, \
                    mis_class_samples = search_res
                if sid not in remained_seq_ids:
                    remained_seq_ids[sid] = [bid]
                else:
                    remained_seq_ids[sid].append(bid)
                # update branch count for samples
                if do_tuning:
                    for spi in selected_target_samples.keys():
                        self.sample2count[matched_class][0][spi] += selected_target_samples[spi]
                    for spi in mis_class_samples.keys():
                        self.sample2count[matched_class][1][spi] += mis_class_samples[spi]
                else:
                    for spi in selected_target_samples:
                        self.sample2count[matched_class][0][spi] += 1
                    for spi in mis_class_samples:
                        self.sample2count[matched_class][1][spi] += 1
                self.sequence_dic[sid].dump_matched_samples(
                    branch_id=bid,
                    threshold=threshold,
                    value_span=value_span,
                    recall=recall,
                    precision=precision,
                    profit=profit,
                    matched_class=matched_class,
                    target_samples=selected_target_samples,
                    mis_class_samples=mis_class_samples
                )
            os.remove(fi)
        for sid in drop_seq_ids.keys():
            self.sequence_dic[sid].drop_branches(branch_ids=drop_seq_ids[sid], block_dic=self.block_dic)
        print('\treserved branches number:', search_branches - drop_count)
        return remained_seq_ids

    def make_sample2weight(self, matched_class):
        """
        :return: sample2weight
        """
        # target samples
        target_branch_count = 0
        all_target_ids = []
        for si in self.sample2count[matched_class][0].keys():
            target_branch_count += self.sample2count[matched_class][0][si]
            all_target_ids.append(si)
        target_branch_count /= len(self.sample2count[matched_class][0])
        selected_target_ids = set(random.sample(all_target_ids, int(len(all_target_ids) / 2)))
        # non target samples
        non_target_branch_count = 0
        all_non_target_ids = []
        for si in self.sample2count[matched_class][1].keys():
            non_target_branch_count += self.sample2count[matched_class][1][si]
            all_non_target_ids.append(si)
        non_target_branch_count /= len(self.sample2count[matched_class][1])
        selected_non_target_ids = set(random.sample(all_non_target_ids, int(len(all_non_target_ids) / 2)))
        # make sample weight
        sample2weight = {}
        pos_weights = []
        for si in self.sample2count[matched_class][0].keys():
            if target_branch_count == 0:
                sample2weight[si] = 1.0
            else:
                weight = target_branch_count - self.sample2count[matched_class][0][si]
                sample2weight[si] = weight
                if weight > 0:
                    pos_weights.append(weight)
        for si in self.sample2count[matched_class][1].keys():
            if non_target_branch_count == 0:
                sample2weight[si] = 1.0
            else:
                weight = non_target_branch_count - self.sample2count[matched_class][1][si]
                sample2weight[si] = weight
                if weight > 0:
                    pos_weights.append(weight)
        if target_branch_count != 0 and non_target_branch_count != 0:
            mean_pos_weight = float(np.mean(np.array(pos_weights)))
            for si in self.sample2count[matched_class][0].keys():
                if si in selected_target_ids:
                    sample2weight[si] += mean_pos_weight
                else:
                    sample2weight[si] -= mean_pos_weight
            for si in self.sample2count[matched_class][1].keys():
                if si in selected_non_target_ids:
                    sample2weight[si] += mean_pos_weight
                else:
                    sample2weight[si] -= mean_pos_weight
        return sample2weight

    def make_sample2count(self, data_path):
        """
        initialize sample2count
        :param data_path:
        :return:
        """
        for i in range(self.num_class):
            self.sample2count[i] = [{}, {}]  # for target class and other class
        data_loader_dev = utils.build_data_loader(
            data_path=data_path,
            target_classes=None,
            num_class=self.num_class,
            repeat=False,
            batch_size=1
        )
        sample_count = 0
        for feat, lab in data_loader_dev:
            label = int(np.squeeze(lab.numpy()))
            for i in range(self.num_class):
                if i == label:
                    self.sample2count[i][0][sample_count] = 0
                else:
                    self.sample2count[i][1][sample_count] = 0
            sample_count += 1
        data_loader_dev.dataset.close_files()
        del data_loader_dev

    def make_match_weight(self):
        """
        make match weight
        :return:
        """
        match_weight = {}
        for cls in self.sample2count.keys():
            branch_count = []
            for spid in self.sample2count[cls][0].keys():
                branch_count.append(self.sample2count[cls][0][spid])
            branch_count = np.array(branch_count, dtype=np.float32)
            mean_value = float(np.mean(branch_count))
            branch_count = mean_value - branch_count
            max_value = float(np.max(branch_count))
            normalizer = np.sum(np.exp(branch_count - max_value))
            match_weight[cls] = {}
            for spid in self.sample2count[cls][0].keys():
                exp_weight = np.exp(
                    (mean_value - self.sample2count[cls][0][spid] - max_value)
                )
                match_weight[cls][spid] = exp_weight / normalizer
        return match_weight

    def eval_branch_drop(self,
                         seq_ids,
                         tune_eval_path,
                         min_precision,
                         batch_size,
                         do_tuning=False):
        """
        do eval and drop branches
        :param seq_ids:
        :param tune_eval_path:
        :param min_precision:
        :param batch_size:
        :param do_tuning:
        :return:
        """
        cur_seq_ids = copy.deepcopy(seq_ids)
        drop_branches = 0
        eval_files = os.listdir(tune_eval_path)
        for fi in eval_files:
            data_file = os.path.join(tune_eval_path, fi)
            # make token2precision or value rate
            tk2stat = {}
            tk2cls = {}
            tk2thd = {}
            for sid in cur_seq_ids:
                for bid in cur_seq_ids[sid]:
                    if 'class_mask' in self.sequence_dic[sid].branch_dic[bid]:
                        token = str(sid) + '_' + str(bid)
                        # [predict, right]
                        tk2stat[token] = [0, 0]
                        target_class = self.sequence_dic[sid].branch_dic[bid]['class_mask'].get_target_class()
                        tk2cls[token] = target_class
                        tk2thd[token] = self.sequence_dic[sid].branch_dic[bid]['out_thd']
                if len(cur_seq_ids[sid]) > 0:
                    self.sequence_dic[sid].set_graph(branch_ids=cur_seq_ids[sid])
            # make dataset
            data_loader = utils.build_data_loader_from_file(
                data_path=data_file,
                repeat=False,
                batch_size=batch_size
            )
            # do forward
            with torch.no_grad():
                for feat, lab in data_loader:
                    total_dic = {}
                    for sid in cur_seq_ids.keys():
                        branch_out = self.sequence_dic[sid].forward(
                            block_dic=self.block_dic,
                            inputs=feat,
                            branch_ids=cur_seq_ids[sid],
                            mode='infer',
                            norm='none',
                            use_bound=True
                        )
                        total_dic.update(branch_out)
                    # count right samples and precision
                    for bi in range(batch_size):
                        true_lab = int(np.squeeze(lab.numpy())[bi])
                        for tk in total_dic.keys():
                            branch_out = float((total_dic[tk][0].clone().detach().numpy())[bi, :])
                            cmp_res = bool((total_dic[tk][1].clone().detach().numpy())[bi])
                            target_class = tk2cls[tk]
                            threshold = tk2thd[tk]
                            if (not cmp_res) and branch_out > threshold:
                                tk2stat[tk][0] += 1
                                if target_class == true_lab:
                                    tk2stat[tk][1] += 1
            del data_loader
            bad_branches = {}
            for tk in tk2stat.keys():
                precision = tk2stat[tk][1] / (tk2stat[tk][0] + 1e-10)
                if precision <= min_precision:
                    sid, bid = tk.split('_')
                    sid = int(sid)
                    bid = int(bid)
                    if sid not in bad_branches:
                        bad_branches[sid] = [bid]
                    else:
                        bad_branches[sid].append(bid)
            # drop bad branches and update sample-branch-count
            for sid in bad_branches.keys():
                to_drop_branches = list(bad_branches[sid])
                drop_branches += len(to_drop_branches)
                for bid in to_drop_branches:
                    matched_class, target_samples, mis_class_samples, profit = \
                        self.sequence_dic[sid].load_matched_samples(branch_id=bid)
                    if do_tuning:
                        for spi in target_samples.keys():
                            self.sample2count[matched_class][0][spi] -= target_samples[spi]
                        for spi in mis_class_samples.keys():
                            self.sample2count[matched_class][1][spi] -= mis_class_samples[spi]
                    else:
                        for spi in target_samples:
                            self.sample2count[matched_class][0][spi] -= 1
                        for spi in mis_class_samples:
                            self.sample2count[matched_class][1][spi] -= 1
                self.sequence_dic[sid].drop_branches(
                    branch_ids=to_drop_branches,
                    block_dic=self.block_dic
                )
            corrected = {}
            for sid in cur_seq_ids.keys():
                for bid in cur_seq_ids[sid]:
                    if sid not in bad_branches or bid not in bad_branches[sid]:
                        if sid not in corrected:
                            corrected[sid] = [bid]
                        else:
                            corrected[sid].append(bid)
            cur_seq_ids = corrected
        return drop_branches, cur_seq_ids

    # part 7: branch tuning
    def select_refine_branches(self, untrained=False):
        """
        select refine branch, to cover full sample set
        :return:
        """
        print('\tstart select refine branches')
        matched_seq = 0
        cls2num = {}
        for i in range(self.num_class):
            cls2num[i] = 0
        selected_seq_ids = {}
        for sid in self.sequence_dic.keys():
            tuning_branches = []
            for bid in self.sequence_dic[sid].branch_dic.keys():
                if 'class_mask' not in self.sequence_dic[sid].branch_dic[bid]:
                    continue
                if untrained:
                    if bid not in self.sequence_dic[sid].trained_mask:
                        tuning_branches.append(bid)
                        target_class = self.sequence_dic[sid].branch_dic[bid]['class_mask'].get_target_class()
                        cls2num[target_class] += 1
                else:
                    tuning_branches.append(bid)
                    target_class = self.sequence_dic[sid].branch_dic[bid]['class_mask'].get_target_class()
                    cls2num[target_class] += 1
            if len(tuning_branches) > 0:
                selected_seq_ids[sid] = copy.deepcopy(tuning_branches)
                matched_seq += len(tuning_branches)
        print('\ttotal tuning branches:', matched_seq)
        for cli in range(self.num_class):
            print('\ttuning branches for class:', cli, 'is', cls2num[cli])
        print('\tselecting done')
        return selected_seq_ids

    def make_initial_train_fixed_output(self,
                                        seq_ids,
                                        data_path,
                                        fixed_output_path,
                                        train_batch_size,
                                        norm='relu'):
        """
        make initial fixed output
        :param seq_ids:
        :param data_path:
        :param fixed_output_path:
        :param train_batch_size:
        :param norm
        :return:
        """
        print('\tstart making initial train fixed output')
        for sid in seq_ids.keys():
            self.sequence_dic[sid].prepare_branch_out(branch_ids=seq_ids[sid])
        for cli in range(self.num_class):
            # build data loader from file
            class_data_file = os.path.join(data_path, str(cli) + '.pkl')
            data_loader = utils.build_data_loader_from_file(
                data_path=class_data_file,
                repeat=False,
                batch_size=train_batch_size
            )
            fixed_out_file = os.path.join(fixed_output_path, str(cli) + '.pkl')
            with open(fixed_out_file, 'wb') as fw:
                with torch.no_grad():
                    for feat, lab in data_loader:
                        total_dic = {}
                        for sid in seq_ids.keys():
                            dump_dic = self.sequence_dic[sid].dump_branch_out(
                                inputs=feat,
                                branch_ids=seq_ids[sid],
                                block_dic=self.block_dic,
                                norm=norm
                            )
                            total_dic.update(dump_dic)
                        fixed_output = np.zeros([train_batch_size, self.num_class], dtype=np.float32)
                        for tk in total_dic.keys():
                            branch_sample_out = total_dic[tk]
                            fixed_output = fixed_output + branch_sample_out
                        if self.additional_bias is not None:
                            fixed_output = fixed_output + self.additional_bias
                        pickle.dump(fixed_output, fw)
        print('\tmake initial train fixed output done')

    def make_initial_fixed_output(self,
                                  seq_ids,
                                  data_path_dic,
                                  batch_size,
                                  norm='relu'):
        """
        make initial fixed output
        :param seq_ids:
        :param data_path_dic:
        :param batch_size:
        :param norm:
        :return:
        """
        print('\tstart making eval and test fixed output')
        for sid in seq_ids.keys():
            self.sequence_dic[sid].prepare_branch_out(branch_ids=seq_ids[sid])
        for data_key in data_path_dic.keys():
            data_loader = utils.build_data_loader(
                data_path=data_path_dic[data_key][0],
                num_class=self.num_class,
                target_classes=None,
                repeat=False,
                batch_size=batch_size
            )
            fixed_output_file = data_path_dic[data_key][1]
            with open(fixed_output_file, 'wb') as fw:
                with torch.no_grad():
                    for feat, lab in data_loader:
                        total_dic = {}
                        for sid in seq_ids.keys():
                            dump_dic = self.sequence_dic[sid].dump_branch_out(
                                inputs=feat,
                                branch_ids=seq_ids[sid],
                                block_dic=self.block_dic,
                                norm=norm
                            )
                            total_dic.update(dump_dic)
                        fixed_output = np.zeros([batch_size, self.num_class], dtype=np.float32)
                        for tk in total_dic.keys():
                            branch_out = total_dic[tk]
                            fixed_output = fixed_output + branch_out
                        if self.additional_bias is not None:
                            fixed_output = fixed_output + self.additional_bias
                        pickle.dump(fixed_output, fw)
            data_loader.dataset.close_files()
            del data_loader
        print('\tmake initial eval and test fixed output done')

    def make_tuning_train_dataset(self,
                                  seq_ids,
                                  data_path,
                                  fixed_output_path,
                                  output_path,
                                  batch_size,
                                  norm='relu'):
        """
        make training dataset in tuning
        :param seq_ids:
        :param data_path:
        :param fixed_output_path:
        :param output_path:
        :param batch_size:
        :param norm:
        :return:
        """
        print('\tstart making tuning train dataset')
        for sid in seq_ids.keys():
            self.sequence_dic[sid].prepare_branch_out(branch_ids=seq_ids[sid])
        for cli in range(self.num_class):
            # build data loader from file
            class_data_file = os.path.join(data_path, str(cli) + '.pkl')
            data_loader = utils.build_data_loader_from_file(
                data_path=class_data_file,
                repeat=False,
                batch_size=batch_size
            )
            class_out_file = os.path.join(output_path, str(cli) + '.pkl')
            fixed_out_file = os.path.join(fixed_output_path, str(cli) + '.pkl')
            fr = open(fixed_out_file, 'rb')
            sample_count = 0
            with open(class_out_file, 'wb') as fw:
                with torch.no_grad():
                    for feat, lab in data_loader:
                        if sample_count % 1000 == 0:
                            print('\t', sample_count)
                        try:
                            fixed_output = pickle.load(fr)
                        except EOFError:
                            print('\tError: there is not enough fixed output')
                            break
                        total_dic = {}
                        for sid in seq_ids.keys():
                            dump_dic = self.sequence_dic[sid].dump_branch_out(
                                inputs=feat,
                                branch_ids=seq_ids[sid],
                                block_dic=self.block_dic,
                                norm=norm
                            )
                            total_dic.update(dump_dic)
                        dump_label = lab.numpy()
                        for bi in range(batch_size):
                            new_dump_dic = {}
                            for tk in total_dic.keys():
                                branch_sample_out = total_dic[tk][bi:bi + 1, :]
                                new_dump_dic[tk] = branch_sample_out
                            pickle.dump([new_dump_dic, fixed_output[bi:bi + 1, :], dump_label[bi:bi + 1]], fw)
                            sample_count += 1
            fr.close()
            del data_loader
        # merge to one training data file
        out_file = os.path.join(
            os.path.dirname(output_path),
            'tuning_data.pkl'
        )
        utils.merge_and_dump_train_data(
            data_path=output_path,
            num_class=self.num_class,
            batch_size=16,
            out_file=out_file
        )
        print('\tmake train dataset done')

    def make_tuning_dataset(self, seq_ids, data_path_dic, norm='relu', batch_size=100):
        """
        make tuning dataset
        :param seq_ids:
        :param data_path_dic:
        :param norm:
        :param batch_size:
        :return:
        """
        print('\tstart making tuning dataset')
        for sid in seq_ids.keys():
            self.sequence_dic[sid].prepare_branch_out(branch_ids=seq_ids[sid])
        for data_key in data_path_dic.keys():
            data_loader = utils.build_data_loader(
                data_path=data_path_dic[data_key][0],
                num_class=self.num_class,
                target_classes=None,
                repeat=False,
                batch_size=batch_size
            )
            fixed_output_path = data_path_dic[data_key][1]
            fr = open(fixed_output_path, 'rb')
            # dump_file = os.path.join(dump_path, 'tuning_' + data_key + '.pkl')
            dump_file = data_path_dic[data_key][2]
            sample_count = 0
            with open(dump_file, 'wb') as fw:
                with torch.no_grad():
                    for feat, lab in data_loader:
                        if sample_count % 50 == 0:
                            print('\t', sample_count)
                        try:
                            fixed_output = pickle.load(fr)
                        except EOFError:
                            print('\tError: there is not enough fixed output')
                            break
                        total_dic = {}
                        for sid in seq_ids.keys():
                            dump_dic = self.sequence_dic[sid].dump_branch_out(
                                inputs=feat,
                                branch_ids=seq_ids[sid],
                                block_dic=self.block_dic,
                                norm=norm
                            )
                            total_dic.update(dump_dic)
                        dump_label = lab.numpy()
                        pickle.dump([total_dic, fixed_output, dump_label], fw)
                        sample_count += 1
            fr.close()
            data_loader.dataset.close_files()
            del data_loader
        print('\tmake dataset done')

    def tune_sequences(self,
                       total_seq_ids,
                       train_steps,
                       learning_rate,
                       batch_size,
                       early_stop,
                       eval_batch_size=100):
        """
        do tuning on class masks
        :param total_seq_ids:
        :param train_steps:
        :param learning_rate:
        :param batch_size:
        :param early_stop:
        :param eval_batch_size:
        :return:
        """
        # get parameters and optimizer
        print('\tstart tuing')
        loss_fn = torch.nn.CrossEntropyLoss()
        # build data
        data_path = os.path.join(self.local_path, 'data', 'train', 'tuning_data.pkl')
        tiny_path = os.path.join(self.local_path, 'data', 'eval', 'tuning_tiny.pkl')
        test_path = os.path.join(self.local_path, 'data', 'test', 'tuning_test.pkl')

        # make reference dev-loss
        eval_res = self.eval_tuning(
            data_path=tiny_path,
            seq_ids={},
            loss_fn=loss_fn,
            eval_batch_size=eval_batch_size
        )
        ref_loss = eval_res[0].clone().detach().numpy()
        print('\treference loss is:', ref_loss)

        data_loader = utils.make_data_iteration_repeat(data_path=data_path, batch_size=batch_size)
        # training
        seq_ids = {}
        all_parameters = [self.train_additional_bias]
        for sid in total_seq_ids.keys():
            tuning_param, tune_branches = self.sequence_dic[sid].prepare_tuning(
                branch_ids=total_seq_ids[sid], use_cuda=self.use_cuda)
            all_parameters += tuning_param
            if len(tune_branches) > 0:
                seq_ids[sid] = tune_branches
        print('\ttuning parameters:', len(all_parameters))
        opt = torch.optim.Adam(params=all_parameters, lr=learning_rate)
        best_loss = None
        non_best = 0
        step = 0
        for data in data_loader:
            token2data, fixed_output, label = data
            label = torch.tensor(label, dtype=torch.long)
            fixed_output = torch.tensor(fixed_output, dtype=torch.float32)
            if self.use_cuda:
                label = label.cuda()
                fixed_output = fixed_output.cuda()
            opt.zero_grad()
            output_dict = {}
            for sid in seq_ids.keys():
                seq_out = self.sequence_dic[sid].tuning_forward(
                    branch_ids=seq_ids[sid],
                    token2data=token2data,
                    use_cuda=self.use_cuda
                )
                output_dict.update(seq_out)
            total_output = None
            for tk in output_dict.keys():
                if total_output is None:
                    total_output = output_dict[tk]
                else:
                    total_output = total_output + output_dict[tk]
            total_output = total_output + fixed_output + self.train_additional_bias
            loss = loss_fn(total_output, label)
            if (step * 10) % self.evaluation_steps == 0:  # for more detailed information
                print('\tLOG train: train loss at step', step, 'is', loss.item())
            loss.backward()
            opt.step()
            # evaluation
            if step % self.evaluation_steps == 0 or step == train_steps - 1:
                tiny_dev = utils.make_iteration(data_path=tiny_path)
                with torch.no_grad():
                    total_loss = None
                    sample_count = 0
                    for dev_data in tiny_dev:
                        dev_token2data, dev_fixed_out, dev_label = dev_data
                        dev_label = torch.tensor(dev_label, dtype=torch.long)
                        dev_fixed_out = torch.tensor(dev_fixed_out, dtype=torch.float32)
                        if self.use_cuda:
                            dev_label = dev_label.cuda()
                            dev_fixed_out = dev_fixed_out.cuda()
                        dev_output_dict = {}
                        for sid in seq_ids.keys():
                            dev_seq_out = self.sequence_dic[sid].tuning_forward(
                                branch_ids=seq_ids[sid],
                                token2data=dev_token2data,
                                use_cuda=self.use_cuda
                            )
                            dev_output_dict.update(dev_seq_out)
                        for j in range(eval_batch_size):
                            dev_total_output = None
                            for tk in dev_output_dict.keys():
                                if dev_total_output is None:
                                    dev_total_output = dev_output_dict[tk][j:j + 1, :]
                                else:
                                    dev_total_output = dev_total_output + dev_output_dict[tk][j:j + 1, :]
                            dev_total_output = \
                                dev_total_output + dev_fixed_out[j:j + 1, :] + self.train_additional_bias
                            dev_loss = loss_fn(dev_total_output, dev_label[j:j + 1])
                            if total_loss is None:
                                total_loss = dev_loss
                            else:
                                total_loss = dev_loss + total_loss
                            sample_count += 1
                    avg_loss = total_loss / sample_count
                    avg_loss = avg_loss.cpu().clone().detach().numpy()
                    if best_loss is None or best_loss > avg_loss:
                        best_loss = avg_loss
                        print('\tsave checkpoint in step:', step)
                        for sid in seq_ids.keys():
                            self.sequence_dic[sid].save_classify_layer(
                                branch_ids=seq_ids[sid], step=step)
                        # save train_additional_bias
                        save_bias_file = os.path.join(self.local_path, 'data', 'additional_bias.pkl')
                        with open(save_bias_file, 'wb') as fw:
                            if self.use_cuda:
                                saved_bias = self.train_additional_bias.clone().detach().cpu()
                            else:
                                saved_bias = self.train_additional_bias.clone().detach()
                            pickle.dump(saved_bias, fw)
                        non_best = 0
                    else:
                        non_best += 1
                print('\tLOG eval: eval loss at step', step, 'is', avg_loss)
                del tiny_dev
            step += 1
            if non_best == early_stop:
                if best_loss is not None and best_loss < ref_loss:
                    break
            if step == train_steps:
                print('\tLOG train: train loss at step', step, 'is', loss.item())
                break
        del data_loader
        num_dropped = 0
        remained_seq_ids = {}
        for sid in seq_ids.keys():
            drop_branches = self.sequence_dic[sid].finish_tuning(
                branch_ids=seq_ids[sid], use_cuda=self.use_cuda)
            if len(drop_branches) > 0:
                self.sequence_dic[sid].drop_branches(branch_ids=drop_branches, block_dic=self.block_dic)
                num_dropped += len(drop_branches)
            for bid in seq_ids[sid]:
                if bid not in drop_branches:
                    if sid not in remained_seq_ids:
                        remained_seq_ids[sid] = [bid]
                    else:
                        remained_seq_ids[sid].append(bid)
        save_bias_file = os.path.join(self.local_path, 'data', 'additional_bias.pkl')
        with open(save_bias_file, 'rb') as fr:
            self.train_additional_bias = pickle.load(fr)
        os.remove(save_bias_file)
        print('\tdropped non-contribution branches:', num_dropped)
        # final evaluation
        eval_res = self.eval_tuning(
            data_path=test_path, 
            seq_ids=remained_seq_ids, 
            loss_fn=loss_fn, 
            eval_batch_size=eval_batch_size
        )
        avg_loss = eval_res[0]
        accuracy = eval_res[1]
        class_count = eval_res[2]
        class_pred = eval_res[3]
        class_recall = eval_res[4]
        print('\ttuning loss:', avg_loss, 'accuracy:', accuracy)
        for k in class_recall.keys():
            print('\tclass:', k, 'recall:',
                  class_recall[k] / class_count[k], 'precision:', class_recall[k] / class_pred[k])
        return avg_loss, accuracy, remained_seq_ids

    def eval_tuning(self, data_path, loss_fn, seq_ids, eval_batch_size):
        """
        eval tuning
        :param data_path:
        :param loss_fn:
        :param seq_ids:
        :param eval_batch_size:
        :return:
        """
        data_loader_dev = utils.make_iteration(data_path=data_path)
        total_loss = None
        sample_count = 0
        right_count = 0
        class_pred = {}
        class_count = {}
        class_recall = {}
        with torch.no_grad():
            for dev_data in data_loader_dev:
                dev_token2data, dev_fixed_out, dev_label = dev_data
                dev_label = torch.tensor(dev_label, dtype=torch.long)
                dev_fixed_out = torch.tensor(dev_fixed_out, dtype=torch.float32)
                dev_output_dict = {}
                for sid in seq_ids.keys():
                    dev_seq_out = self.sequence_dic[sid].tuning_forward(
                        branch_ids=seq_ids[sid],
                        token2data=dev_token2data,
                        use_cuda=False
                    )
                    dev_output_dict.update(dev_seq_out)
                for j in range(eval_batch_size):
                    true_label = np.squeeze(dev_label.numpy()).astype(np.int32)[j]
                    if int(true_label) not in class_count:
                        class_count[int(true_label)] = 1
                    else:
                        class_count[int(true_label)] += 1
                    dev_total_output = None
                    for tk in dev_output_dict.keys():
                        if dev_total_output is None:
                            dev_total_output = dev_output_dict[tk][j:j + 1, :]
                        else:
                            dev_total_output = dev_total_output + dev_output_dict[tk][j:j + 1, :]
                    if dev_total_output is not None:
                        dev_total_output = dev_total_output + dev_fixed_out[j:j + 1, :] + self.train_additional_bias
                    else:
                        dev_total_output = dev_fixed_out[j:j + 1, :]
                    dev_loss = loss_fn(dev_total_output, dev_label[j:j + 1])
                    if total_loss is None:
                        total_loss = dev_loss
                    else:
                        total_loss = dev_loss + total_loss
                    prediction = np.argmax(np.squeeze(dev_total_output.clone().detach().numpy()))
                    if int(prediction) not in class_pred:
                        class_pred[int(prediction)] = 1
                    else:
                        class_pred[int(prediction)] += 1
                    if prediction == true_label:
                        if int(prediction) not in class_recall:
                            class_recall[int(prediction)] = 1
                        else:
                            class_recall[int(prediction)] += 1
                        right_count += 1
                    sample_count += 1
        avg_loss = total_loss / sample_count
        accuracy = right_count / sample_count
        return avg_loss, accuracy, class_count, class_pred, class_recall

    def merge_train_fixed_data(self, seq_ids, data_path, fixed_output_path, batch_size):
        """
        merge data to fixed output
        :param seq_ids:
        :param data_path:
        :param fixed_output_path:
        :param batch_size:
        :return:
        """
        for sid in seq_ids.keys():
            self.sequence_dic[sid].set_graph(branch_ids=seq_ids[sid])
        for cli in range(self.num_class):
            data_file = os.path.join(data_path, str(cli) + '.pkl')
            data_loader = utils.make_iteration(data_path=data_file)
            out_file = os.path.join(fixed_output_path, str(cli) + '.pkl')
            count = 0
            fixed_outputs = []
            with open(out_file, 'wb') as fw:
                with torch.no_grad():
                    for train_data in data_loader:
                        token2data, fixed_out, label = train_data
                        output_dict = {}
                        for sid in seq_ids.keys():
                            seq_out = self.sequence_dic[sid].tuning_forward(
                                branch_ids=seq_ids[sid],
                                token2data=token2data,
                                use_cuda=False
                            )
                            output_dict.update(seq_out)
                        merged_fixed_out = fixed_out
                        for tk in output_dict.keys():
                            merged_fixed_out = merged_fixed_out + output_dict[tk].clone().detach().numpy()
                        merged_fixed_out = merged_fixed_out + self.train_additional_bias
                        fixed_outputs.append(copy.deepcopy(merged_fixed_out))
                        count += 1
                        if count != 0 and count % batch_size == 0:
                            total_fixed_outputs = np.concatenate(fixed_outputs, axis=0)
                            pickle.dump(total_fixed_outputs, fw)
                            fixed_outputs = []
            del data_loader

    def merge_fixed_data(self, seq_ids, eval_file, eval_fixed_file, test_file, test_fixed_file):
        """
        merge data to fixed output
        :param seq_ids:
        :param eval_file:
        :param eval_fixed_file:
        :param test_file:
        :param test_fixed_file:
        :return:
        """
        for sid in seq_ids.keys():
            self.sequence_dic[sid].set_graph(branch_ids=seq_ids[sid])
        data_list = [[eval_file, eval_fixed_file], [test_file, test_fixed_file]]
        for dai in data_list:
            data_loader_eval = utils.make_iteration(data_path=dai[0])
            with open(dai[1], 'wb') as fw:
                with torch.no_grad():
                    for eval_data in data_loader_eval:
                        eval_token2data, eval_fixed_out, eval_label = eval_data
                        output_dict = {}
                        for sid in seq_ids.keys():
                            seq_out = self.sequence_dic[sid].tuning_forward(
                                branch_ids=seq_ids[sid],
                                token2data=eval_token2data,
                                use_cuda=False
                            )
                            output_dict.update(seq_out)
                        for tk in output_dict.keys():
                            eval_fixed_out = eval_fixed_out + output_dict[tk].clone().detach().numpy()
                        eval_fixed_out = eval_fixed_out + self.train_additional_bias
                        pickle.dump(eval_fixed_out, fw)
            del data_loader_eval

    def tuning_step(self, last_step, data_path, tiny_path, test_path, match_train_path):
        """
        do tuning
        :return:
        """
        # make data path
        dump_path = os.path.join(self.local_path, 'data')
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        # for training
        dump_train_path = os.path.join(self.local_path, 'data', 'train')
        if not os.path.exists(dump_train_path):
            os.makedirs(dump_train_path)
        train_fixed_output_path = os.path.join(dump_train_path, 'fixed_output')
        if not os.path.exists(train_fixed_output_path):
            os.makedirs(train_fixed_output_path)
        train_data_path = os.path.join(dump_train_path, 'train_data')
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
        # for eval and test
        dump_eval_path = os.path.join(self.local_path, 'data', 'eval')
        if not os.path.exists(dump_eval_path):
            os.makedirs(dump_eval_path)
        dump_test_path = os.path.join(self.local_path, 'data', 'test')
        if not os.path.exists(dump_test_path):
            os.makedirs(dump_test_path)
        # make tuning dataset
        tune_seq_ids = self.select_refine_branches(untrained=True)  # only untrained branches is selected
        num_tokens = 0
        for sid in tune_seq_ids:
            num_tokens += len(tune_seq_ids[sid])
        print('\tnumber of tokens:', num_tokens)
        eval_fixed_output_file = os.path.join(self.local_path, 'data', 'eval', 'fixed_output.pkl')
        test_fixed_output_file = os.path.join(self.local_path, 'data', 'test', 'fixed_output.pkl')
        if (not os.path.exists(eval_fixed_output_file)) or (not os.path.exists(test_fixed_output_file)):
            fixed_seq_ids = {}
            for sid in self.sequence_dic.keys():
                fixed_seq_ids[sid] = list(self.sequence_dic[sid].trained_mask)
            self.make_initial_train_fixed_output(
                seq_ids=fixed_seq_ids,
                data_path=data_path,
                fixed_output_path=train_fixed_output_path,
                train_batch_size=100,
                norm='none'
            )
            fixed_data_path_dic = {
                'eval': [tiny_path, eval_fixed_output_file],
                'test': [test_path, test_fixed_output_file]
            }
            self.make_initial_fixed_output(
                seq_ids=fixed_seq_ids,
                data_path_dic=fixed_data_path_dic,
                batch_size=100,
                norm='none'
            )
        self.make_tuning_train_dataset(
            seq_ids=tune_seq_ids,
            data_path=data_path,
            fixed_output_path=train_fixed_output_path,
            output_path=train_data_path,
            batch_size=100,
            norm='none'
        )
        data_path_dic = {
            'tiny': [
                tiny_path,
                eval_fixed_output_file,
                os.path.join(self.local_path, 'data', 'eval', 'tuning_tiny.pkl')
            ],
            'test': [
                test_path,
                test_fixed_output_file,
                os.path.join(self.local_path, 'data', 'test', 'tuning_test.pkl')
            ]
        }
        self.make_tuning_dataset(
            seq_ids=tune_seq_ids,
            data_path_dic=data_path_dic,
            norm='none',
            batch_size=100
        )
        # do tuning
        self.adjust_bias()
        self.evaluation_steps = 100
        best_loss, accuracy, remained_seq_ids = self.tune_sequences(
            total_seq_ids=tune_seq_ids,
            train_steps=30000,
            learning_rate=self.learning_rate * 4,
            batch_size=8,
            early_stop=20
        )
        self.train_additional_bias = self.train_additional_bias.clone().detach().numpy()
        # merge new data to fixed output
        print('\tstart merge fixed data')
        self.merge_train_fixed_data(
            seq_ids=remained_seq_ids,
            data_path=train_data_path,
            fixed_output_path=train_fixed_output_path,
            batch_size=100
        )
        self.merge_fixed_data(
            seq_ids=remained_seq_ids,
            eval_file=os.path.join(self.local_path, 'data', 'eval', 'tuning_tiny.pkl'),
            eval_fixed_file=eval_fixed_output_file,
            test_file=os.path.join(self.local_path, 'data', 'test', 'tuning_test.pkl'),
            test_fixed_file=test_fixed_output_file
        )
        if self.additional_bias is None:
            self.additional_bias = self.train_additional_bias
        else:
            self.additional_bias = self.additional_bias + self.train_additional_bias
        print('\tmerge fixed data done')
        if last_step is not None:
            save_step = last_step
        else:
            save_step = utils.extract_number(os.path.basename(self.model_path))
        self.make_compensation_base(
            data_path=match_train_path, step=last_step, batch_size=50, seq_ids=remained_seq_ids)
        model_path = os.path.join(self.local_path, 'model_tune' + str(save_step) + '.pkl')
        param_path = os.path.join(self.local_path, 'param_tune' + str(save_step) + '.pkl')
        self.save(model_path=model_path, param_path=param_path)
        return best_loss, accuracy, remained_seq_ids

    def adjust_bias(self):
        """
        adjust bias
        :return:
        """
        additional_bias = np.zeros([self.num_class], dtype=np.float32)
        for cls in self.sample2count.keys():
            all_target_values = []
            for spi in self.sample2count[cls][0].keys():
                all_target_values.append(self.sample2count[cls][0][spi])
            target_mean = np.mean(np.array(all_target_values))
            additional_bias[cls] = -target_mean
        if self.use_cuda:
            self.train_additional_bias = torch.nn.Parameter(
                torch.tensor(additional_bias, dtype=torch.float32).cuda()
            )
        else:
            self.train_additional_bias = torch.nn.Parameter(torch.tensor(additional_bias, dtype=torch.float32))

    def test_refine_branch(self,
                           eval_path,
                           seq_ids,
                           batch_size=50,
                           stat_dump_file=None,
                           log=True,
                           from_file=False,
                           cls2cnt=None,
                           cls_cnt_std=None):
        """
        test refine branches
        :param eval_path:
        :param seq_ids:
        :param batch_size:
        :param stat_dump_file:
        :param log:
        :param from_file:
        :param cls2cnt:
        :param cls_cnt_std:
        :return:
        """
        if not from_file:
            data_loader_dev = utils.build_data_loader(
                data_path=eval_path,
                target_classes=None,
                num_class=self.num_class,
                repeat=False,
                batch_size=batch_size
            )
        else:
            data_loader_dev = utils.build_data_loader_from_file(
                data_path=eval_path,
                batch_size=batch_size,
                repeat=False
            )
        cls2tk = {}
        tk2thd = {}
        tk2stat = {}
        for sid in seq_ids.keys():
            for bid in seq_ids[sid]:
                token = str(sid) + '_' + str(bid)
                self.sequence_dic[sid].branch_dic[bid]['class_mask'].disable()
                refine_class = self.sequence_dic[sid].branch_dic[bid]['class_mask'].get_target_class()
                if refine_class not in cls2tk:
                    cls2tk[refine_class] = [token]
                else:
                    cls2tk[refine_class].append(token)
                tk2thd[token] = self.sequence_dic[sid].branch_dic[bid]['out_thd']
                tk2stat[token] = [0, 0]  # right, predict
        if log:
            print('\tnumber of tokens:', len(tk2thd))
        test_seq_ids = {}
        for sid in seq_ids.keys():
            for bid in seq_ids[sid]:
                if bid == 0 and sid < self.tree_volume:
                    continue
                if sid not in test_seq_ids:
                    test_seq_ids[sid] = [bid]
                else:
                    test_seq_ids[sid].append(bid)
            if sid in test_seq_ids:
                self.sequence_dic[sid].set_graph(branch_ids=test_seq_ids[sid])
        dump_file = os.path.join(self.local_path, 'test_refine.pkl')
        sample_count = 0
        cls_sp_cnt = {}
        with open(dump_file, 'wb') as fw:
            for feat_dev, lab_dev in data_loader_dev:
                total_out_dic = {}
                for sid in test_seq_ids.keys():
                    dev_class_output = self.sequence_dic[sid].forward(
                        block_dic=self.block_dic,
                        inputs=feat_dev,
                        branch_ids=test_seq_ids[sid],
                        mode='infer',
                        norm='none',
                        set_bound=False,
                        use_bound=True
                    )
                    total_out_dic.update(dev_class_output)
                for i in range(batch_size):
                    out_dic = {}
                    true_lab = int(np.squeeze(lab_dev.numpy())[i])
                    if true_lab not in cls_sp_cnt:
                        cls_sp_cnt[true_lab] = 1
                    else:
                        cls_sp_cnt[true_lab] += 1
                    for tk in total_out_dic.keys():
                        branch_out = float(np.squeeze(total_out_dic[tk][0].clone().detach().numpy())[i])
                        cmp_res = bool(np.squeeze(total_out_dic[tk][1].clone().detach().numpy())[i])
                        thd = tk2thd[tk]
                        if not cmp_res:
                            if branch_out >= thd:
                                out_dic[tk] = branch_out
                            else:
                                out_dic[tk] = None
                    pickle.dump([sample_count, out_dic, true_lab], fw)
                    sample_count += 1
        if not from_file:
            data_loader_dev.dataset.close_files()
        del data_loader_dev
        if cls2cnt is None and cls_cnt_std is None:
            cls2cnt = {}
            cls2sqcnt = {}
            sample_count = 0
            with open(dump_file, 'rb') as fr:
                while True:
                    try:
                        sp = pickle.load(fr)
                        sp_id, out_dic, label = sp
                        count = 0
                        if label not in cls2tk:
                            continue
                        for tk in cls2tk[label]:
                            if tk not in out_dic or out_dic[tk] is None:
                                continue
                            else:
                                count += 1
                        if label not in cls2cnt:
                            cls2cnt[label] = count
                        else:
                            cls2cnt[label] += count
                        if label not in cls2sqcnt:
                            cls2sqcnt[label] = count ** 2
                        else:
                            cls2sqcnt[label] += count ** 2
                        sample_count += 1
                    except EOFError:
                        break
            cls_cnt_std = {}
            for cls in cls2cnt.keys():
                cls2cnt[cls] /= cls_sp_cnt[cls]
                cls2sqcnt[cls] /= cls_sp_cnt[cls]
                cls_cnt_std[cls] = float(np.sqrt(cls2sqcnt[cls] - (cls2cnt[cls] ** 2)))
        clspred = {}
        cls2right = {}
        with open(dump_file, 'rb') as fr:
            while True:
                try:
                    sp = pickle.load(fr)
                    sp_id, out_dic, label = sp
                    cls_score = {}
                    for cls in cls2tk.keys():
                        for tk in cls2tk[cls]:
                            if tk not in out_dic:
                                continue
                            if out_dic[tk] is None:
                                score = 0
                            else:
                                score = 1.0
                                tk2stat[tk][1] += 1
                                if cls == label:
                                    tk2stat[tk][0] += 1
                            if cls not in cls_score:
                                cls_score[cls] = score
                            else:
                                cls_score[cls] += score
                    pred_lab = None
                    max_score = None
                    for cli in cls_score.keys():
                        cls_score[cli] = (cls_score[cli] - cls2cnt[cli]) / (cls_cnt_std[cli] + 1e-10)
                        if max_score is None or max_score < cls_score[cli]:
                            max_score = cls_score[cli]
                            pred_lab = cli
                    if pred_lab not in clspred:
                        clspred[pred_lab] = 1
                    else:
                        clspred[pred_lab] += 1
                    if pred_lab == label:
                        if pred_lab not in cls2right:
                            cls2right[pred_lab] = 1
                        else:
                            cls2right[pred_lab] += 1
                except EOFError:
                    break
        total_right = 0
        total_sample = 0
        for cls in cls2tk.keys():
            if cls not in clspred:
                pred_cnt = 0
            else:
                pred_cnt = clspred[cls]
            if cls not in cls2right:
                right_cnt = 0
            else:
                right_cnt = cls2right[cls]
            total_sample += pred_cnt
            total_right += right_cnt
            if log:
                print('\ttest class:', cls,
                      'precision:', (right_cnt / pred_cnt),
                      'recall:', (right_cnt / cls_sp_cnt[cls]))
        if log:
            print('\ttotal accuracy:', total_right / total_sample)
        # dump token stat
        if stat_dump_file is not None:
            with open(stat_dump_file, 'w', encoding='utf8') as fw:
                for cls in cls2tk.keys():
                    for tk in cls2tk[cls]:
                        recall = tk2stat[tk][0] / cls_sp_cnt[cls]
                        precision = tk2stat[tk][0] / tk2stat[tk][1]
                        fw.write(tk + '\t' + str(recall) + '\t' + str(precision) + '\t' + str(cls) + '\n')
        return total_right / total_sample, cls2cnt, cls_cnt_std

    # part 8: reinforcement loop
    def circular_management(self,
                            use_steps,
                            data_path,
                            match_train_path,
                            tiny_path,
                            test_path,
                            data_dump_path,
                            tune_eval_path,
                            batch_size,
                            cluster_steps=1,
                            step='ori',
                            end_step='end',
                            make_initial=False):
        """
        do dynamic network training, do match each explore steps
        :param use_steps:
        :param data_path:
        :param match_train_path:
        :param tiny_path:
        :param test_path:
        :param data_dump_path:
        :param tune_eval_path:
        :param batch_size:
        :param step: start step selection:
        :param end_step: last step of training:
        :param cluster_steps:
        :param make_initial:
        :return:
        """
        # explore phase
        best_loss = None
        false_rate = None
        if step == 'ori' and end_step != 'ori':
            data_loader = utils.build_data_loader(
                data_path=data_path, num_class=self.num_class, target_classes=None, repeat=True, batch_size=batch_size)
            data_loader_dev = utils.build_data_loader(
                data_path=test_path, num_class=self.num_class, target_classes=None, repeat=False, batch_size=1)
            tiny_dev = utils.build_data_loader(
                data_path=tiny_path, num_class=self.num_class, target_classes=None, repeat=False, batch_size=100)
            best_loss, false_rate = self.train_sequences(
                data_loader=data_loader,
                data_loader_dev=data_loader_dev,
                seq_ids=None,
                train_steps=self.max_train_steps,
                learning_rate=self.learning_rate,
                tiny_dev=tiny_dev,
                include_branch=True,
                norm='none',
                joint_training=True,
                eval_batch_size=100
            )
            model_path = os.path.join(self.local_path, 'model_ori.pkl')
            param_path = os.path.join(self.local_path, 'param_ori.pkl')
            self.save(model_path=model_path, param_path=param_path)
            print('=== train sequences done === step: ori')
            print('best loss for original training is:', best_loss)
            print('best false rate for original training is:', false_rate)
            data_loader.dataset.close_files()
            data_loader_dev.dataset.close_files()
            tiny_dev.dataset.close_files()
            del data_loader
            del data_loader_dev
            del tiny_dev
            gc.collect()
            step = 'clt'
        if step == 'clt':
            if end_step == 'clt':
                return best_loss, false_rate
            data_loader_dev = utils.build_data_loader(
                data_path=match_train_path,
                num_class=self.num_class,
                target_classes=None,
                repeat=False,
                batch_size=1
            )
            # get input range
            self.get_input_range(data_loader_dev=data_loader_dev)
            for i in range(cluster_steps):
                # do reference and target cluster
                self.dump_and_cluster(seq_ids=None)
                self.ref_cluster_dump(data_loader_dev=data_loader_dev, seq_ids=None)
                print('clustering done step: ', i, '/', cluster_steps)
            data_loader_dev.dataset.close_files()
            gc.collect()
            for sid in self.sequence_dic.keys():
                self.sequence_dic[sid].update_clusters()
            model_path = os.path.join(self.local_path, 'model_clt.pkl')
            param_path = os.path.join(self.local_path, 'param_clt.pkl')
            self.save(model_path=model_path, param_path=param_path)
            step = 'uda'
        if step == 'uda':
            if end_step == 'uda':
                return best_loss, false_rate
            if len(self.retrieval_files) == 0:
                for sid in self.sequence_dic.keys():
                    retrieval_file = os.path.join(self.sequence_dic[sid].local_path, 'retrieval.pkl')
                    if os.path.exists(retrieval_file):
                        self.retrieval_files.add(retrieval_file)
            start_step = utils.extract_number(os.path.basename(self.model_path))
            print('start step is:', start_step, self.model_path)
            match_data_path = os.path.join(self.local_path, 'data')
            if not os.path.exists(match_data_path):
                os.makedirs(match_data_path)
            if start_step == 0:
                self.make_sample2count(data_path=match_train_path)
            sample2count_file = os.path.join(self.local_path, 'data', 'sample2count.pkl')
            if make_initial and (not os.path.exists(sample2count_file)):
                self.make_compensation_base(
                    data_path=match_train_path, step=start_step, batch_size=50, seq_ids=None)
            self.add_rate = 0.1
            for sid in self.sequence_dic.keys():
                self.sequence_dic[sid].update_clusters()
            # randomly select reference clusters
            all_tgt_cluster_files = []
            for sid in self.sequence_dic.keys():
                for ti in self.sequence_dic[sid].target_clusters:
                    all_tgt_cluster_files.append(ti)
            all_ref_cluster_files = [os.path.join(data_dump_path, fi) for fi in os.listdir(data_dump_path)]
            print('reference clusters:', len(all_ref_cluster_files), 'target clusters:', len(all_tgt_cluster_files))
            # do matching
            # to_match = int(self.tree_volume * self.add_rate)
            to_match = 1
            print('num to match:', to_match)
            to_use = to_match * 8
            search_batch_size = 1000
            data_loader_dev = utils.build_data_loader(
                data_path=match_train_path,
                target_classes=None,
                num_class=self.num_class,
                repeat=False,
                batch_size=search_batch_size
            )
            for i in range(use_steps):
                random.shuffle(all_ref_cluster_files)
                self.match_branches_data(
                    data_files=all_ref_cluster_files,
                    to_match=to_match,
                    target_clusters=all_tgt_cluster_files,
                    sub_sample=int(len(all_tgt_cluster_files) * 0.1)
                )
                new_branches = {}
                for cli in range(self.num_class):
                    print('\tuse branches on class:', cli)
                    use_count = 1
                    while use_count > 0:
                        # do match and tune
                        class_new_branches, use_count = self.use_branches_data(
                            to_use=to_use,
                            refine_class=cli
                        )
                        if use_count > 0:
                            remained_seq_ids = self.search_branch_results(
                                out_dir='match',
                                data_loader_dev=data_loader_dev,
                                seq_ids=class_new_branches,
                                matched_class=cli,
                                min_precision=0.11,
                                min_recall=0.03,
                                batch_size=search_batch_size,
                                tune_eval_path=tune_eval_path,
                                do_tuning=make_initial
                            )
                            for sid in remained_seq_ids.keys():
                                if sid not in new_branches:
                                    new_branches[sid] = []
                                for bid in remained_seq_ids[sid]:
                                    new_branches[sid].append(bid)
                print('=== matching done === step:', i)
                eval_drop_num, eval_remained_seq_ids = self.eval_branch_drop(
                    seq_ids=new_branches,
                    tune_eval_path=tune_eval_path,
                    min_precision=0.11,
                    batch_size=1000,
                    do_tuning=make_initial
                )
                print('new branches is:', eval_remained_seq_ids)
                if (i + 1) % 50 == 0 or i == use_steps - 1:
                    model_path = os.path.join(self.local_path, 'model_match' + str(i + start_step + 1) + '.pkl')
                    param_path = os.path.join(self.local_path, 'param_match' + str(i + start_step + 1) + '.pkl')
                    self.save(model_path=model_path, param_path=param_path)
                    last_step = i + start_step + 1
                    if make_initial:
                        best_loss, accuracy, tune_seq_ids = self.tuning_step(
                            last_step=last_step,
                            data_path=data_path,
                            tiny_path=tiny_path,
                            test_path=test_path,
                            match_train_path=match_train_path
                        )
                    else:
                        print('\t === start test added branches ===')
                        tune_seq_ids = self.select_refine_branches()
                        cur_seq_ids = copy.deepcopy(tune_seq_ids)
                        accuracy, cls2cnt, cls_cnt_std = self.test_refine_branch(
                            eval_path=match_train_path,
                            seq_ids=cur_seq_ids,
                            stat_dump_file=os.path.join(self.local_path, 'token_stat_eval.txt'),
                            cls2cnt=None,
                            cls_cnt_std=None,
                            from_file=False
                        )
                        self.test_refine_branch(
                            eval_path=tiny_path,
                            seq_ids=cur_seq_ids,
                            stat_dump_file=os.path.join(self.local_path, 'token_stat_tiny.txt'),
                            cls2cnt=cls2cnt,
                            cls_cnt_std=cls_cnt_std,
                            from_file=True
                        )
            data_loader_dev.dataset.close_files()
            del data_loader_dev
        return best_loss, false_rate

    # part 8: save and load
    def save(self, model_path, param_path):
        """
        save checkpoint of current step
        :param model_path: model parameters
        :param param_path: controller parameter path
        :return:
        """
        with open(model_path, 'wb') as fw:
            pickle.dump(self.block_dic, fw)
            pickle.dump(self.sequence_dic, fw)
            pickle.dump(self.additional_bias, fw)
            pickle.dump(self.train_additional_bias, fw)
        with open(param_path, 'wb') as fw:
            pickle.dump(self.threads, fw)
            # structure parameters
            pickle.dump(self.tree_volume, fw)
            pickle.dump(self.kernel_dim, fw)
            pickle.dump(self.max_layers, fw)
            pickle.dump(self.input_dim, fw)
            pickle.dump(self.use_cuda, fw)
            pickle.dump(self.num_class, fw)
            # model parameter
            pickle.dump(self.block_id, fw)
            pickle.dump(self.seq_id, fw)
            # train parameters
            pickle.dump(self.learning_rate, fw)
            pickle.dump(self.max_train_steps, fw)
            pickle.dump(self.early_stop, fw)
            pickle.dump(self.evaluation_steps, fw)
            # reinforcement control parameters
            pickle.dump(self.add_rate, fw)
            pickle.dump(self.cluster_rate, fw)
            pickle.dump(self.dump_rate, fw)
            # for matching
            pickle.dump(self.matched_pairs, fw)
            pickle.dump(self.used_pairs, fw)
            pickle.dump(self.dist_pairs, fw)
            pickle.dump(self.retrieval_files, fw)
            pickle.dump(self.node_pairs, fw)
            # for data matching
            pickle.dump(self.data_matched_pairs, fw)
            pickle.dump(self.data_used_pairs, fw)
            pickle.dump(self.data_dist_pairs, fw)
            pickle.dump(self.data_node_pairs, fw)
            pickle.dump(self.range2sid, fw)
            pickle.dump(self.sample2count, fw)

    def load(self, model_path, param_path):
        """
        load data from checkpoint
        :param model_path:
        :param param_path: controller parameter path
        :return:
        """
        if model_path is not None:
            with open(model_path, 'rb') as fr:
                loaded_block_dic = pickle.load(fr)
                for bli in loaded_block_dic.keys():
                    self.block_dic[bli] = parameter_tree_cm.ParameterBlock(
                        kernel_dim=self.kernel_dim,
                        parameter_block=loaded_block_dic[bli]
                    )
                loaded_sequence_dic = pickle.load(fr)
                for sid in loaded_sequence_dic.keys():
                    self.sequence_dic[sid] = parameter_tree_cm.ParameterTree(
                        seq_id=sid,
                        num_class=self.num_class,
                        kernel_dim=self.kernel_dim,
                        local_path=os.path.join(self.local_path, 'sequences', str(sid)),
                        dim_range=[],
                        parameter_tree=loaded_sequence_dic[sid],
                        channel=None
                    )
                try:
                    self.additional_bias = pickle.load(fr)
                except EOFError:
                    self.additional_bias = None
                try:
                    self.train_additional_bias = pickle.load(fr)
                except EOFError:
                    self.train_additional_bias = None
        if param_path is not None:
            with open(param_path, 'rb') as fr:
                self.threads = pickle.load(fr)
                # structure parameters
                self.tree_volume = pickle.load(fr)
                self.kernel_dim = pickle.load(fr)
                self.max_layers = pickle.load(fr)
                self.input_dim = pickle.load(fr)
                self.use_cuda = pickle.load(fr)
                self.num_class = pickle.load(fr)
                # model parameter
                self.block_id = pickle.load(fr)
                self.seq_id = pickle.load(fr)
                # train parameters
                self.learning_rate = pickle.load(fr)
                self.max_train_steps = pickle.load(fr)
                self.early_stop = pickle.load(fr)
                self.evaluation_steps = pickle.load(fr)
                # reinforcement control parameters
                self.add_rate = pickle.load(fr)
                self.cluster_rate = pickle.load(fr)
                self.dump_rate = pickle.load(fr)
                # for matching
                self.matched_pairs = pickle.load(fr)
                self.used_pairs = pickle.load(fr)
                self.dist_pairs = pickle.load(fr)
                self.retrieval_files = pickle.load(fr)
                self.node_pairs = pickle.load(fr)
                # for data matching
                self.data_matched_pairs = pickle.load(fr)
                self.data_used_pairs = pickle.load(fr)
                self.data_dist_pairs = pickle.load(fr)
                self.data_node_pairs = pickle.load(fr)
                try:
                    self.range2sid = pickle.load(fr)
                except EOFError:
                    self.range2sid = {}
                try:
                    self.sample2count = pickle.load(fr)
                except EOFError:
                    self.sample2count = {}

    # part x: other temp functions
    def make_compensation_base(self, data_path, step, seq_ids, batch_size=50):
        """
        make compensation base
        :param data_path:
        :param batch_size:
        :param step:
        :param seq_ids:
        :return:
        """
        # make initial data match pairs and range2sid
        total_tokens = 0
        if step == 0:
            seq_ids = {}
            total_tokens = len(self.sequence_dic)
            for sid in self.sequence_dic.keys():
                channel = self.sequence_dic[sid].channel
                dim_range = self.sequence_dic[sid].dim_range
                ref_node = '_'.join([str(channel), str(dim_range[0]), str(dim_range[2])])
                self.range2sid[ref_node] = sid
                init_sequence = self.sequence_dic[sid].branch_dic[0]['sequence']
                target_node_key = str(sid) + '-0_' + str(init_sequence[0])
                ref_node_key = '_1-' + ref_node
                if ref_node_key not in self.data_node_pairs:
                    self.data_node_pairs[ref_node_key] = set()
                self.data_node_pairs[ref_node_key].add(target_node_key)
                seq_ids[sid] = [0]
                self.sequence_dic[sid].set_graph(branch_ids=seq_ids[sid])
        else:
            if seq_ids is not None:
                for sid in seq_ids.keys():
                    # clear trained mask for making initial sample2count
                    self.sequence_dic[sid].clear_trained_mask(branch_ids=seq_ids[sid])
            else:
                seq_ids = {}
                for sid in self.sequence_dic.keys():
                    self.sequence_dic[sid].clear_trained_mask(branch_ids=None)
                    seq_ids[sid] = list(self.sequence_dic[sid].branch_dic.keys())
        print('\ttotal number of tokens is:', total_tokens)
        # build data loader
        if step == 0:
            data_loader_dev = utils.build_data_loader(
                data_path=data_path,
                target_classes=None,
                num_class=self.num_class,
                repeat=False,
                batch_size=batch_size
            )
            data_path = os.path.join(self.local_path, 'data')
            if not os.path.exists(data_path):
                os.makedirs(data_path)
        else:
            match_train_path = os.path.join(self.local_path, 'data', 'tuning_match_train.pkl')
            temp_fixed_out = os.path.join(self.local_path, 'data', 'temp_fixed_out.pkl')
            if not os.path.exists(temp_fixed_out):
                batch_size = 100
                match_train_files = [os.path.join(data_path, fi) for fi in os.listdir(data_path)]
                match_sample_count = 0
                for mfi in match_train_files:
                    with open(mfi, 'rb') as fr:
                        while True:
                            try:
                                pickle.load(fr)
                                match_sample_count += 1
                            except EOFError:
                                break
                dump_num = int(match_sample_count / batch_size)
                print('\tmatch train sample count:', match_sample_count, 'dump number:', dump_num)
                with open(temp_fixed_out, 'wb') as fw:
                    for ri in range(dump_num):
                        pickle.dump(0, fw)
            data_path_dic = {
                'match_train': [data_path, temp_fixed_out, match_train_path]}
            self.make_tuning_dataset(
                seq_ids=seq_ids,
                data_path_dic=data_path_dic,
                norm='none'
            )
            data_loader_dev = utils.make_iteration(data_path=match_train_path)
        # make initial sample2count
        if step == 0:
            restart = False
        else:
            restart = True
            batch_size = 100
        self.make_initial_sample2count(
            data_loader_dev=data_loader_dev, 
            seq_ids=seq_ids, 
            batch_size=batch_size, 
            restart=restart
        )
        if step == 0:
            data_loader_dev.dataset.close_files()
        for sid in self.sequence_dic.keys():
            self.sequence_dic[sid].set_trained_mask()
        del data_loader_dev

    def make_initial_sample2count(self, data_loader_dev, seq_ids, batch_size, restart):
        """
        make initial sample count
        :param data_loader_dev:
        :param seq_ids:
        :param batch_size:
        :param restart:
        :return:
        """
        tk2cls = {}
        if restart:
            # load sample2count
            sample2count_file = os.path.join(self.local_path, 'data', 'sample2count.pkl')
            if os.path.exists(sample2count_file):
                with open(sample2count_file, 'rb') as fr:
                    self.sample2count = pickle.load(fr)
            else:
                for cls in range(self.num_class):
                    for sp in self.sample2count[cls][0].keys():
                        self.sample2count[cls][0][sp] = 0
                    for sp in self.sample2count[cls][1].keys():
                        self.sample2count[cls][1][sp] = 0
            # make tk2cls
            for sid in seq_ids:
                for bid in seq_ids[sid]:
                    if 'class_mask' in self.sequence_dic[sid].branch_dic[bid]:
                        self.sequence_dic[sid].branch_dic[bid]['class_mask'].enable()
                        target_class = self.sequence_dic[sid].branch_dic[bid]['class_mask'].get_target_class()
                        token = str(sid) + '_' + str(bid)
                        tk2cls[token] = target_class
        # do forward
        sample_count = 0
        for data in data_loader_dev:
            total_out_dic = {}
            if restart:
                feat_dev, fixed_out, lab_dev = data
            else:
                feat_dev, lab_dev = data
            for sid in seq_ids.keys():
                if not restart:
                    dev_class_output = self.sequence_dic[sid].forward(
                        block_dic=self.block_dic,
                        inputs=feat_dev,
                        branch_ids=seq_ids[sid],
                        mode='infer',
                        set_bound=False,
                        use_bound=False,
                        norm='relu'
                    )
                else:
                    dev_class_output = self.sequence_dic[sid].tuning_forward(
                        branch_ids=seq_ids[sid],
                        token2data=feat_dev,
                        use_cuda=False
                    )
                total_out_dic.update(dev_class_output)
            for i in range(batch_size):
                # compute mean value of each sample
                if not restart:
                    true_lab = int(np.squeeze(lab_dev.numpy())[i])
                else:
                    true_lab = int(lab_dev[i])
                # update sample2count
                for tk in total_out_dic.keys():
                    if tk not in tk2cls:
                        branch_out = (total_out_dic[tk].clone().detach().numpy()[i, :]).astype(np.float32)
                        for j in range(self.num_class):
                            if j == true_lab:
                                self.sample2count[j][0][sample_count] += float(branch_out[j])
                            else:
                                self.sample2count[j][1][sample_count] += float(branch_out[j])
                    else:
                        branch_out = (total_out_dic[tk].clone().detach().numpy()[i, :]).astype(np.float32)
                        target_class = tk2cls[tk]
                        if target_class == true_lab:
                            self.sample2count[target_class][0][sample_count] += float(branch_out[target_class])
                        else:
                            self.sample2count[target_class][1][sample_count] += float(branch_out[target_class])
                sample_count += 1
        if self.train_additional_bias is not None:
            for cls in range(self.num_class):
                for si in self.sample2count[cls][0].keys():
                    self.sample2count[cls][0][si] += float(self.train_additional_bias[cls])
                for si in self.sample2count[cls][1].keys():
                    self.sample2count[cls][1][si] += float(self.train_additional_bias[cls])
        # dump sample2count
        sample2count_file = os.path.join(self.local_path, 'data', 'sample2count.pkl')
        with open(sample2count_file, 'wb') as fw:
            pickle.dump(self.sample2count, fw)

    def dump_branch_output(self, match_train_path):
        """
        dump one output of branch on uniform distribution
        :return:
        """
        data_loader_dev = utils.build_data_loader(
            data_path=match_train_path,
            num_class=self.num_class,
            target_classes=None,
            repeat=False,
            batch_size=1
        )
        # get input range
        self.get_input_range(data_loader_dev=data_loader_dev)
        self.dump_and_cluster(seq_ids={48: [0]})
        data_loader_dev.dataset.close_files()
        gc.collect()
