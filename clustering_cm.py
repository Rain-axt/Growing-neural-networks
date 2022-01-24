# -*-coding:utf8-*-

import math
import numpy as np
import random
import os
from multiprocessing import Pool
import gc
import pickle
import bisect
import copy


MIN_DISTANCE = 0.000001


# part 1: basic functions
def euclidean_dist(point_a, point_b, weight=None):
    """
    compute weighted euclidean dist
    :param point_a:
    :param point_b:
    :param weight:
    :return:
    """
    if weight is None:
        total = np.sum((point_a - point_b) ** 2, axis=-1)
    else:
        total = np.sum(((point_a - point_b) * weight) ** 2, axis=-1)
    return np.sqrt(total)


def gaussian_kernel(distance, bandwidth):
    """
    compute gaussian probability here
    :param distance:
    :param bandwidth: kernel width of gaussian kernel
    :return:
    """
    euclidean_distance = np.sqrt((distance ** 2).sum(axis=1))
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (euclidean_distance / bandwidth) ** 2)
    return val


def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = bandwidths.size

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val


# part 2: mean shift cluster
def densest_point(kernel, points, kernel_bandwidth, iteration_callback=None):
    """
    find the densest location in the space, each time find one location
    :param kernel: here, we use Gaussian kernel
    :param points: list of point, each point is a np.array
    :param kernel_bandwidth: covariance of Gaussian kernel, default is 1 after BatchNorm
    :param iteration_callback:
    :return: the densest location
    """
    if iteration_callback:
        iteration_callback(points, 0)
    shift_center = random.choice(points)
    max_min_dist = 1
    iteration_number = 0

    still_shifting = True
    # the min shift distance is specified
    while max_min_dist > MIN_DISTANCE:
        # print max_min_dist
        max_min_dist = 0
        iteration_number += 1
        if not still_shifting:
            break
        p_new_start = shift_center
        p_new = shift_point(kernel, shift_center, points, kernel_bandwidth)
        dist = euclidean_dist(p_new, p_new_start)
        if dist > max_min_dist:
            max_min_dist = dist
        if dist < MIN_DISTANCE:
            still_shifting = False
        shift_center = p_new
    return shift_center


def shift_point(kernel, point, points, kernel_bandwidth):
    # from http://en.wikipedia.org/wiki/Mean-shift
    points = np.array(points)

    # numerator
    point_weights = kernel(point - points, kernel_bandwidth)
    tiled_weights = np.tile(point_weights, [point.size, 1])
    # denominator
    denominator = np.sum(point_weights)
    shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
    return shifted_point


def nearest_distance(point, points):
    """
    find nearest points around point
    :param point: single point
    :param points: list of point
    :return: list of nearest points
    """
    dif = point - np.array(points)
    dists = np.sqrt(np.sum(dif ** 2, axis=1))
    sorted_dist = np.sort(dists)
    if len(points) > 1:
        min_dist = sorted_dist[1]
    else:
        min_dist = sorted_dist[0]
    return min_dist


def nearest_center_distance(point, clusters):
    """
    find nearest center
    :param point:
    :param clusters:
    :return:
    """
    centers = np.array([ci['center'] for ci in clusters])
    dif = point - centers
    dists = np.sqrt(np.sum(dif ** 2, axis=1))
    min_dist = np.min(dists)
    min_id = np.argmin(dists)
    return min_dist, min_id


def cluster_by_neighbor_distance(center, max_dist, points, id_set, min_dist_rate=0.2):
    """
    find all the points within the max-neighbor-distance
    use mean-distance of positive and negative as boundary
    :param center: center of cluster
    :param max_dist: maximum neighbor distance
    :param points: list of points
    :param id_set: set of point ids
    :param min_dist_rate: minimum dist count for adding this point to group
    :return: dict of {center, upper-bound, lower-bound}
    """
    group = set()
    # make initial members
    dist_count = 0
    for pid in id_set:
        point = points[pid]
        dist = euclidean_dist(center, point)
        if dist <= max_dist:
            group.add(pid)
            dist_count += 1
    new_member = len(group)
    min_dist_count = dist_count * min_dist_rate
    while new_member > 0:
        new_member = 0
        for pid in id_set:
            dist_count = 0
            if pid in group:
                continue
            for si in group:
                point = points[pid]
                dist = euclidean_dist(point, points[si])
                if dist <= max_dist:
                    dist_count += 1
            if dist_count >= min_dist_count:
                group.add(pid)
                new_member += 1
    if len(group) <= 1:
        # no point included in this group
        return group, None
    # compute upper-bound and lower-bound for each model
    point_group = np.zeros([len(group), center.shape[0]], dtype=np.float32)
    for count, pid in enumerate(group):
        point_group[count] += points[pid]
    new_center = np.mean(point_group, axis=0)
    upper_bound = np.max(point_group, axis=0)
    lower_bound = np.min(point_group, axis=0)
    cluster_info = {
        'center': new_center,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'samples': group  # retain point ids of this cluster
    }
    return group, cluster_info


def merge_to_cluster(point, pid, cluster_infos):
    """
    merge one point ot cluster-info list
    :param point:
    :param pid:
    :param cluster_infos:
    :return:
    """
    min_center_distance, min_id = nearest_center_distance(point=point, clusters=cluster_infos)
    min_id = int(min_id)
    ori_sample_count = len(cluster_infos[min_id]['samples'])
    merged_mean = (cluster_infos[min_id]['center'] * ori_sample_count + point) \
        / (ori_sample_count + 1)
    cluster_infos[min_id]['center'] = merged_mean
    cluster_infos[min_id]['samples'].add(pid)
    ori_upper = cluster_infos[min_id]['upper_bound']
    ori_lower = cluster_infos[min_id]['lower_bound']
    merged_upper = np.maximum(ori_upper, point)
    merged_lower = np.minimum(ori_lower, point)
    cluster_infos[min_id]['upper_bound'] = merged_upper
    cluster_infos[min_id]['lower_bound'] = merged_lower
    return cluster_infos


def mean_shift_cluster(points,
                       kernel=multivariate_gaussian_kernel,
                       width=0.1):
    """
    do mean shift cluster
    :param points: a list of points
    :param kernel: kernel function to compute shift
    :param width: bandwidth of gaussian kernel
    :return: a list of cluster-info
    """
    current_pid_set = set()
    for i in range(len(points)):
        current_pid_set.add(i)

    # prepare band width for kernel, here, we assume that each dimension is independent
    band_width = np.ones(shape=[points[0].shape[0]], dtype=np.float32) * width

    remove_pids = set()
    cluster_infos = []
    while len(current_pid_set) > 0:
        # do mean-shift-cluster on the remained points
        if len(remove_pids) > 0:
            for ri in remove_pids:
                current_pid_set.remove(ri)
            if len(current_pid_set) == 0:
                break
            remains = []
            for pid in current_pid_set:
                remains.append(points[pid])
        else:
            remains = points
        densest = densest_point(kernel=kernel, kernel_bandwidth=band_width, points=remains)
        min_point_distance = float(nearest_distance(point=densest, points=remains))
        if len(cluster_infos) > 0:
            min_center_distance, min_id = nearest_center_distance(point=densest, clusters=cluster_infos)
            min_center_distance = float(min_center_distance)
        else:
            min_center_distance = min_point_distance
        if min_center_distance < min_point_distance or len(current_pid_set) == 1:
            # the case that center part of remained samples is clustered
            # randomly choose one sample and merge into nearest cluster
            chosen_id = random.choice(list(current_pid_set))
            cluster_infos = merge_to_cluster(
                point=points[chosen_id], pid=chosen_id, cluster_infos=cluster_infos)
            remove_pids = set()
            remove_pids.add(chosen_id)
        else:
            remove_pids, cluster_info = cluster_by_neighbor_distance(
                center=densest,
                max_dist=min(max(min_point_distance, width), min_center_distance),
                points=points,
                id_set=current_pid_set
            )
            if cluster_info is not None:
                cluster_infos.append(cluster_info)
            else:
                # merge single point into exist clusters
                if len(remove_pids) == 1:
                    chosen_id = list(remove_pids)[0]
                else:
                    chosen_id = random.choice(list(current_pid_set))
                    remove_pids = set()
                    remove_pids.add(chosen_id)
                cluster_infos = merge_to_cluster(
                    point=points[chosen_id], pid=chosen_id, cluster_infos=cluster_infos)
    return cluster_infos


def class_cluster_statistic(clusters, dim):
    """
    do center, scale and dim sort
    :param clusters: a list of cluster info
    :param dim:
    :return:
    """
    mean = np.zeros([dim], dtype=np.float32)
    total_samples = 0
    boundaries = []
    for ci in clusters:
        total_samples += len(ci['samples'])
        boundaries.append(ci['lower_bound'])
        boundaries.append(ci['upper_bound'])
    for ci in clusters:
        mean += ((len(ci['samples']) / total_samples) * ci['center'])
    boundaries = np.array(boundaries)
    total_upper = np.max(boundaries, axis=0)
    total_lower = np.min(boundaries, axis=0)
    std = total_upper - total_lower
    # sort dims
    sorted_dim_orders = np.zeros(shape=[dim, dim], dtype=np.float32)
    # shape is [sorted_arg, origin_dims]
    for ci in clusters:
        sorted_arg = np.argsort(ci['center'])
        for j in range(dim):
            sorted_dim_orders[j, sorted_arg[j]] += len(ci['samples'])
    sorted_dims = []
    dim_set = set()
    for j in range(dim):
        dims = np.argsort(sorted_dim_orders[j])
        for i in range(dims.size - 1, -1, -1):
            if dims[i] not in dim_set:
                dim_set.add(dims[i])
                sorted_dims.append(dims[i])
                break
    sorted_dims = np.array(sorted_dims)
    new_clusters = []
    for ci in clusters:
        new_ci = {
            'center': ((ci['center'] - mean) / (std + 1e-10))[sorted_dims],
            'upper_bound': ((ci['upper_bound'] - mean) / (std + 1e-10))[sorted_dims],
            'lower_bound': ((ci['lower_bound'] - mean) / (std + 1e-10))[sorted_dims],
            'samples': ci['samples']
        }
        if 'max_act' in ci:
            new_ci['max_act'] = ci['max_act']
        new_clusters.append(new_ci)
    # mean and std remain original dim order
    return mean, std, sorted_dims, new_clusters


def do_cluster(arguments):
    """
    :param arguments: input_file, output_file, nearest_count, distance_scale
    :return:
    """
    infile, outfile, width, compute_stat, ref = arguments
    print('\tdoing cluster on dump file:', infile)
    points = {}  # for different label, do cluster separately
    total_points = []
    class_sample_ids = {}
    id2act_value = {}
    with open(infile, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                if ref:
                    sample_id, point, label = sp
                else:
                    sample_id, point, label, act_value = sp
                    id2act_value[sample_id] = act_value
                total_points.append(point)
                if not isinstance(label, int):
                    label = int(label.item())
                if label in points:
                    points[label].append(point)
                else:
                    points[label] = [point]
                if label not in class_sample_ids:  # map class-sample-id to global-sample-id
                    class_sample_ids[label] = {len(points[label]) - 1: sample_id}
                else:
                    class_sample_ids[label][len(points[label]) - 1] = sample_id
            except EOFError:
                break
    if len(total_points) == 0:
        print('\tno points in file:', infile)
        return
    kernel_dim = total_points[0].size
    class_clusters = {}
    # do cluster
    for li in points.keys():
        # add manual normalization for all the points
        normed_points = []
        points_mean = np.mean(np.array(points[li]), axis=0)
        upper_bound = np.max(np.array(points[li]), axis=0)
        lower_bound = np.min(np.array(points[li]), axis=0)
        points_std = upper_bound - lower_bound  # change std to value range
        for pi in points[li]:
            normed_point = (pi - points_mean) / (points_std + 1e-10)
            normed_points.append(normed_point)
        cluster_infos = mean_shift_cluster(
            points=normed_points,
            width=width
        )
        for idc, ci in enumerate(cluster_infos):
            # de-normalize centers and boundaries
            normed_center = cluster_infos[idc]['center']
            denorm_center = normed_center * (points_std + 1e-10) + points_mean
            cluster_infos[idc]['center'] = denorm_center
            normed_upper_bound = cluster_infos[idc]['upper_bound']
            denorm_upper_bound = normed_upper_bound * (points_std + 1e-10) + points_mean
            cluster_infos[idc]['upper_bound'] = denorm_upper_bound
            normed_lower_bound = cluster_infos[idc]['lower_bound']
            denorm_lower_bound = normed_lower_bound * (points_std + 1e-10) + points_mean
            cluster_infos[idc]['lower_bound'] = denorm_lower_bound
            # change samples to global sample-id
            new_samples = set()
            for si in cluster_infos[idc]['samples']:
                new_samples.add(class_sample_ids[li][si])
            cluster_infos[idc]['samples'] = new_samples
            if not ref:
                # use the max-value as the center of this cluster
                max_act = None
                max_spi = None
                for spi in cluster_infos[idc]['samples']:
                    act_value = id2act_value[spi]
                    if max_act is None or max_act < act_value:
                        max_act = act_value
                        max_spi = spi
                for ids in class_sample_ids[li]:
                    if class_sample_ids[li][ids] == max_spi:
                        max_act_point = points[li][ids]
                        cluster_infos[idc]['center'] = max_act_point
                        cluster_infos[idc]['max_act'] = max_act
                        break
        class_clusters[li] = cluster_infos
    # compute statistic and dump
    cluster_count = 0
    with open(outfile, 'wb') as fw:
        # dump format: [cluster-info, cluster-statistic, label, cluster-id]
        for li in class_clusters.keys():
            if len(class_clusters[li]) == 0:
                continue
            if compute_stat:
                mean, std, sorted_dims, new_clusters = class_cluster_statistic(
                    clusters=class_clusters[li], dim=kernel_dim)
            else:
                mean = None
                std = None
                sorted_dims = None
                new_clusters = class_clusters[li]
            for idx, ni in enumerate(new_clusters):
                new_cluster = copy.deepcopy(ni)
                new_cluster['cluster_id'] = cluster_count  # add cluster-id in cluster
                dump_cluster = [
                    new_cluster,
                    [mean, std, sorted_dims],
                    li
                ]
                cluster_count += 1
                pickle.dump(dump_cluster, fw)


def parallel_cluster(input_files, threads, out_dirs, width=0.1, compute_stat=True, ref=False):
    """
    do cluster in parallel
    :param input_files:
    :param threads:
    :param out_dirs:
    :param width:
    :param compute_stat:
    :param ref:
    :return:
    """
    arg_list = []
    out_files = []
    for idf, fi in enumerate(input_files):
        outfile = os.path.join(out_dirs[idf], os.path.basename(fi))
        out_files.append(outfile)
        if os.path.exists(outfile):
            continue
        arg_list.append([fi, outfile, width, compute_stat, ref])

    pool = Pool(processes=threads)
    pool.map(do_cluster, arg_list)
    pool.close()
    pool.join()
    gc.collect()
    return out_files


def compute_overlap(ref_upper, ref_lower, target_upper, target_lower):
    """
    compute overlap rates, compute relative overlap rate
    :param ref_upper: absolute reference upper bound, shape as [kernel_dims]
    :param ref_lower: absolute reference lower bound, shape as [kernel_dims]
    :param target_upper: absolute target upper bound, shape as [kernel_dims]
    :param target_lower: absolute target lower bound, shape as [kernel_dims]
    :return: overlap rate
    """
    ref_ovp_rates = []
    kernel_dim = ref_lower.size
    ovp_upper = np.zeros([kernel_dim], dtype=np.float32)
    ovp_lower = np.zeros([kernel_dim], dtype=np.float32)
    for i in range(ref_upper.size):
        max_ub = max(ref_upper[i], target_upper[i])
        min_lb = min(ref_lower[i], target_lower[i])
        total = max_ub - min_lb
        ref_span = ref_upper[i] - ref_lower[i]
        # overlap length
        ovp = ref_upper[i] - ref_lower[i] + target_upper[i] - target_lower[i] - total
        ovp = max(ovp, 0)
        sorted_bound = np.sort(np.array([ref_lower[i], ref_upper[i], target_lower[i], target_upper[i]]))
        ovp_lower[i] = sorted_bound[1]
        ovp_upper[i] = sorted_bound[2]
        ref_ovp_rates.append(ovp / (ref_span + 1e-10))
    if min(ref_ovp_rates) == 0:
        return 0, None, None  # no overlap here
    else:
        # use log to ensure data stability
        ref_log_sum = np.sum(np.log(np.array(ref_ovp_rates)))
        # compute root by log-division
        ref_avg_rate = np.exp(ref_log_sum / ref_upper.size)
        return ref_avg_rate, ovp_lower, ovp_upper


def overlap_recluster(arguments):
    """
    compute overlap and re-cluster for reference clusters
    :return:
    """
    cluster_file, dump_file, kernel_dim, num_class, out_file, max_overlap, width = arguments
    print('\tdo recluster in cluster file:', cluster_file)
    # load points from dump_file, used for further clusters
    class_points = {}
    with open(dump_file, 'rb') as fr:
        while True:
            try:
                point = pickle.load(fr)
                label = int(point[2])
                if label not in class_points:
                    class_points[label] = [point]
                else:
                    class_points[label].append(point)
            except EOFError:
                break
    # load clusters
    class_dic = {}
    with open(cluster_file, 'rb') as fr:
        while True:
            try:
                dump_cluster = pickle.load(fr)
                cluster, stats, label = dump_cluster
                if label not in class_dic:
                    class_dic[label] = [cluster]
                else:
                    class_dic[label].append(cluster)
            except EOFError:
                break
    # compute overlap and re-cluster
    count = 0
    cls2idx = {}
    all_clusters = []
    for target_label in range(num_class):
        if target_label not in class_dic:
            # just in case
            continue
        out_clusters = []
        # select bad clusters
        for tci in class_dic[target_label]:
            bad_samples = set()
            for li in class_dic.keys():
                if int(li) == int(target_label):
                    continue
                for rci in class_dic[li]:
                    # compute overlap
                    ovp_rate, ovp_lower, ovp_upper = compute_overlap(
                        ref_lower=tci['lower_bound'],
                        ref_upper=tci['upper_bound'],
                        target_lower=rci['lower_bound'],
                        target_upper=rci['upper_bound']
                    )
                    if ovp_rate > 0:
                        # count bad samples in overlap range
                        for pi in class_points[target_label]:
                            if pi[0] not in tci['samples']:
                                continue
                            point = pi[1]
                            if np.all(point <= ovp_upper) and np.all(point >= ovp_lower):
                                bad_samples.add(pi[0])
            max_ovp = len(bad_samples) / len(tci['samples'])
            if max_ovp <= max_overlap:
                out_clusters.append(tci)
                continue
            recluster_ids = []
            recluster_points = []
            for pi in class_points[target_label]:
                if pi[0] in tci['samples'] and pi[0] not in bad_samples:
                    recluster_points.append(pi[1])
                    recluster_ids.append(pi[0])
            if len(recluster_ids) <= 1:
                continue
            # add manual normalization and de-normalization
            normed_points = []
            point_array = np.array(recluster_points)
            points_mean = np.mean(point_array, axis=0)
            points_std = np.std(point_array, axis=0)
            for pi in recluster_points:
                normed_point = (pi - points_mean) / (points_std + 1e-10)
                normed_points.append(normed_point)
            new_clusters = mean_shift_cluster(points=normed_points, width=width)
            # map sample id of bad samples to global samples
            for idc, ni in enumerate(new_clusters):
                # de-normalize centers and boundaries
                normed_center = new_clusters[idc]['center']
                denorm_center = normed_center * (points_std + 1e-10) + points_mean
                new_clusters[idc]['center'] = denorm_center
                normed_upper_bound = new_clusters[idc]['upper_bound']
                denorm_upper_bound = normed_upper_bound * (points_std + 1e-10) + points_mean
                new_clusters[idc]['upper_bound'] = denorm_upper_bound
                normed_lower_bound = new_clusters[idc]['lower_bound']
                denorm_lower_bound = normed_lower_bound * (points_std + 1e-10) + points_mean
                new_clusters[idc]['lower_bound'] = denorm_lower_bound
                new_samples = set()
                for si in ni['samples']:
                    # transfer sample-id of re-cluster-ids to global-ids
                    new_samples.add(recluster_ids[si])
                new_clusters[idc]['samples'] = new_samples
                out_clusters.append(new_clusters[idc])
        if len(out_clusters) == 0:
            continue
        cls2idx[target_label] = []
        for ci in out_clusters:
            all_clusters.append(ci)
            cls2idx[target_label].append(count)
            count += 1
    cluster_count = 0
    with open(out_file, 'wb') as fw:
        for li in cls2idx.keys():
            class_clusters = []
            for idx in cls2idx[li]:
                class_clusters.append(all_clusters[idx])
            if len(class_clusters) == 0:
                continue
            mean, std, sorted_dims, new_clusters = class_cluster_statistic(
                clusters=class_clusters, dim=kernel_dim)
            for ni in new_clusters:
                new_cluster = copy.deepcopy(ni)
                new_cluster['cluster_id'] = cluster_count  # add cluster-id in cluster
                to_dump = [
                    new_cluster,
                    [mean, std, sorted_dims],
                    li
                ]
                pickle.dump(to_dump, fw)
                cluster_count += 1


def parallel_recluster(cluster_files,
                       dump_files,
                       kernel_dim,
                       num_class,
                       out_files,
                       threads,
                       max_overlap=0.5,
                       width=0.1):
    """
    do re-cluster in parallel
    :param cluster_files:
    :param dump_files:
    :param kernel_dim:
    :param num_class:
    :param out_files:
    :param threads:
    :param max_overlap:
    :param width:
    :return:
    """
    arg_list = []
    for idx, fi in enumerate(cluster_files):
        arg_list.append([fi, dump_files[idx], kernel_dim, num_class, out_files[idx], max_overlap, width])
    # parallel re-cluster
    pool = Pool(processes=threads)
    pool.map(overlap_recluster, arg_list)
    pool.close()
    pool.join()


# part3 retrieval support
def update_clusters(retrieval_file, input_files):
    """
    update retrieval file to incorporate new clusters
    :param retrieval_file:
    :param input_files:
    :return:
    """
    # load query data-set
    # file format
    # 1. first-value list
    # 2. last-value list
    # 3. cluster-information [sequence-id, node, label, cluster-id]
    if os.path.exists(retrieval_file):
        with open(retrieval_file, 'rb') as fr:
            first_values = pickle.load(fr)
            last_values = pickle.load(fr)
            first_cluster_infos = pickle.load(fr)
            last_cluster_infos = pickle.load(fr)
    else:
        first_values = []
        last_values = []
        first_cluster_infos = []
        last_cluster_infos = []
    # load new clusters
    for fi in input_files:
        with open(fi, 'rb') as fr:
            node = os.path.basename(fi).split('.')[0]
            sid = int(fi.split('/')[-3])
            while True:
                try:
                    dump_cluster = pickle.load(fr)
                    center = dump_cluster[0]['center']
                    first_value = center[0]
                    last_value = center[-1]
                    cluster_id = dump_cluster[0]['cluster_id']
                    label = dump_cluster[2]
                    first_index = bisect.bisect(first_values, first_value)
                    last_index = bisect.bisect(last_values, last_value)
                    bisect.insort(first_values, first_value)
                    bisect.insort(last_values, last_value)
                    new_info = [sid, node, label, cluster_id]
                    first_cluster_infos.insert(first_index, new_info)
                    last_cluster_infos.insert(last_index, new_info)
                except EOFError:
                    break
    # save retrieval info
    with open(retrieval_file, 'wb') as fw:
        pickle.dump(first_values, fw)
        pickle.dump(last_values, fw)
        pickle.dump(first_cluster_infos, fw)
        pickle.dump(last_cluster_infos, fw)


def retrieve_cluster(arguments):
    """
    retrieve matched target clusters, and return matched cluster-infos
    :param arguments
    :return: candidate nodes and labels, used to do matching
    """
    ref_clusters, retrieval_file, ref_sid, ref_node, beam, matched_pairs, to_match_label, match_data = arguments
    # match-key: reference sequence-id_branch-id_block-id_label
    # match-value: a set {target sequence-id_branch-id_block-id_label}
    ref_key = str(ref_sid) + '-' + ref_node + '-' + str(to_match_label)
    candidate_nodes = set()
    # load cluster information
    with open(retrieval_file, 'rb') as fr:
        first_values = pickle.load(fr)
        last_values = pickle.load(fr)
        first_cluster_infos = pickle.load(fr)
        last_cluster_infos = pickle.load(fr)
    # retrieve candidate branches
    for ri in ref_clusters:
        first_center = ri['center'][0]
        last_center = ri['center'][-1]
        first_index = bisect.bisect(first_values, first_center)
        first_start = max(first_index - beam, 0)
        first_end = min(first_index + beam, len(first_values))
        last_index = bisect.bisect(last_values, last_center)
        last_start = max(last_index - beam, 0)
        last_end = min(last_index + beam, len(last_values))
        # count candidate nodes
        for fi in range(first_start, first_end):
            sid, node, label, cluster_id = first_cluster_infos[fi]
            flg_cnt = False
            if not match_data:
                if int(sid) == int(ref_sid) and node == ref_node and int(label) == int(to_match_label):
                    flg_cnt = True
            target_key = str(sid) + '-' + node + '-' + str(label)
            if ref_key in matched_pairs and target_key in matched_pairs[ref_key]:
                flg_cnt = True
            if flg_cnt:
                continue
            node_info = str(sid) + '+' + str(node) + '+' + str(label)
            candidate_nodes.add(node_info)
        for li in range(last_start, last_end):
            sid, node, label, cluster_id = last_cluster_infos[li]
            flg_cnt = False
            if not match_data:
                if int(sid) == int(ref_sid) and node == ref_node and int(label) == int(to_match_label):
                    flg_cnt = True
            target_key = str(sid) + '-' + node + '-' + str(label)
            if ref_key in matched_pairs and target_key in matched_pairs[ref_key]:
                flg_cnt = True
            if flg_cnt:
                continue
            node_info = str(sid) + '+' + str(node) + '+' + str(label)
            candidate_nodes.add(node_info)
    return candidate_nodes


def parallel_retrieve(ref_clusters,
                      retrieval_files,
                      ref_sid,
                      ref_node,
                      beam,
                      matched_pairs,
                      to_match_label,
                      threads,
                      match_data=False):
    """
    retrieve clusters in parallel
    :param ref_clusters:
    :param retrieval_files:
    :param ref_sid:
    :param ref_node:
    :param beam:
    :param matched_pairs:
    :param to_match_label:
    :param threads:
    :param match_data:
    :return:
    """
    arg_list = []
    for rfi in retrieval_files:
        arg_list.append([ref_clusters, rfi, ref_sid, ref_node, beam, matched_pairs, to_match_label, match_data])
    retrieval_results = set()
    # parallel retrieve
    pool = Pool(processes=threads)
    res_list = pool.map(retrieve_cluster, arg_list)
    pool.close()
    pool.join()
    gc.collect()
    # merge results
    for ri in res_list:
        retrieval_results = retrieval_results.union(ri)
    return retrieval_results


# part3 match functions
def match_clusters_weighted_dist(arguments):
    """
    do cluster matching and return match rate
    using weighted sum of cluster-center-distance
    arguments: ref_clusters, candidate_node, candidate_label
    ref_clusters:
    candidate_node: node file contains node clusters
    candidate_label: retrieved label
    :return:
    """
    ref_node, candidate_node, num_class = arguments

    # load reference clusters
    cls2ref_clt = {}
    with open(ref_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2ref_clt:
                    cls2ref_clt[int(label)] = []
                cls2ref_clt[int(label)].append(cluster)
            except EOFError:
                break
    # load target clusters
    cls2tgt_clt = {}
    with open(candidate_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2tgt_clt:
                    cls2tgt_clt[int(label)] = []
                cls2tgt_clt[int(label)].append(cluster)
            except EOFError:
                break

    distance_array = np.zeros([num_class, num_class], dtype=np.float32) - 1
    # match distance
    for rli in cls2ref_clt.keys():
        ref_sample_count = 0
        for ri in cls2ref_clt[rli]:
            ref_sample_count += len(ri['samples'])
        for tli in cls2tgt_clt.keys():
            tgt_sample_count = 0
            for ti in cls2tgt_clt[tli]:
                tgt_sample_count += len(ti['samples'])
            tgt_clusters = cls2tgt_clt[tli]
            total_center_dist = 0
            for ri in cls2ref_clt[rli]:
                tgt_dist = 0
                for idt, ti in enumerate(tgt_clusters):
                    tgt_dist += (euclidean_dist(ri['center'], ti['center'])) * len(ti['samples']) / tgt_sample_count
                total_center_dist += tgt_dist * len(ri['samples']) / ref_sample_count
            distance_array[int(rli), int(tli)] = total_center_dist

    # find a fitted-list for every reference class for
    matched_dict = {}  # key: ref-label, value: [matched-target-classes]
    dist_dict = {}  # key: ref-label+target-label, value: match-distance
    for tli in cls2tgt_clt.keys():
        dist_res = distance_array[:, tli]
        max_value = np.max(dist_res)
        if max_value == -1:
            continue
        for i in range(num_class):
            if dist_res[i] < 0:
                dist_res[i] = max_value + 1
        matched_ref_label = int(np.argmin(dist_res))
        if matched_ref_label not in matched_dict:
            matched_dict[matched_ref_label] = [tli]
        else:
            matched_dict[matched_ref_label].append(tli)
        dist_key = str(matched_ref_label) + '+' + str(tli)
        dist_dict[dist_key] = np.min(dist_res)

    return matched_dict, dist_dict, candidate_node


def cluster_restat(ref_clusters, ref_stat, target_clusters, target_stat, out_file, target_rate):
    """
    do counter-stat and re-stat for matched clusters
    return transferred stat and transferred distance
    :param ref_clusters:
    :param ref_stat:
    :param target_clusters:
    :param target_stat:
    :param out_file:
    :param target_rate:
    :return:
    """
    # do counter stat
    new_ref_clusters = []
    ref_mean, ref_std, ref_dim_order = ref_stat
    kernel_dim = ref_mean.size
    ref_transfer_order = np.zeros([kernel_dim], dtype=np.int32)
    for di in range(kernel_dim):
        trans_dim = ref_dim_order[di]
        ref_transfer_order[trans_dim] = di
    matched_ref_samples = set()
    for rci in ref_clusters:
        new_cluster = {
            'center': rci['center'][ref_transfer_order] * (ref_std + 1e-10) + ref_mean,
            'upper_bound': rci['upper_bound'][ref_transfer_order] * (ref_std + 1e-10) + ref_mean,
            'lower_bound': rci['lower_bound'][ref_transfer_order] * (ref_std + 1e-10) + ref_mean,
            'samples': rci['samples']
        }
        new_ref_clusters.append(new_cluster)
        matched_ref_samples = matched_ref_samples.union(rci['samples'])
    # do re-stat
    new_ref_mean, new_ref_std, new_ref_sorted_dims, _ = class_cluster_statistic(
        clusters=new_ref_clusters, dim=kernel_dim)
    # do counter stat
    new_tgt_clusters = []
    tgt_mean, tgt_std, tgt_dim_order = target_stat
    tgt_transfer_order = np.zeros([kernel_dim], dtype=np.int32)
    for di in range(kernel_dim):
        trans_dim = tgt_dim_order[di]
        tgt_transfer_order[trans_dim] = di
    for tci in target_clusters:
        new_cluster = {
            'center': tci['center'][tgt_transfer_order] * (tgt_std + 1e-10) + tgt_mean,
            'upper_bound': tci['upper_bound'][tgt_transfer_order] * (tgt_std + 1e-10) + tgt_mean,
            'lower_bound': tci['lower_bound'][tgt_transfer_order] * (tgt_std + 1e-10) + tgt_mean,
            'samples': tci['samples']
        }
        new_tgt_clusters.append(new_cluster)
    # do re-stat
    new_tgt_mean, new_tgt_std, new_tgt_sorted_dims, transfer_tgt_clusters = class_cluster_statistic(
        clusters=new_tgt_clusters, dim=kernel_dim)
    # compute transferred distance
    scale_factor = np.zeros([kernel_dim], dtype=np.float32)
    for i in range(kernel_dim):
        tgt_dim = new_tgt_sorted_dims[i]
        ref_dim = new_ref_sorted_dims[i]
        scale = new_tgt_std[tgt_dim] / new_ref_std[ref_dim]
        scale_factor[i] = scale
    if np.any(np.isinf(scale_factor)) or np.any(np.isnan(scale_factor)):
        return None
    # dump stat
    with open(out_file, 'wb') as fw:
        pickle.dump([new_ref_mean, new_ref_std, new_ref_sorted_dims], fw)
        pickle.dump([new_tgt_mean, new_tgt_std, new_tgt_sorted_dims], fw)
        pickle.dump(target_rate, fw)
        pickle.dump(matched_ref_samples, fw)
    total_dist = 0
    for idt, tci in enumerate(target_clusters):
        dist = euclidean_dist(tci['center'], ref_clusters[idt]['center'], weight=scale_factor)
        # dist = euclidean_dist(tci['center'], ref_clusters[idt]['center'], weight=None)
        total_dist += dist
    avg_dist = total_dist / len(target_clusters)
    return avg_dist


def dump_match_information(ref_stat,
                           target_stat,
                           out_file,
                           boundary=None):
    """
    dump matching information
    :param ref_stat:
    :param target_stat:
    :param out_file:
    :param boundary:
    :return:
    """
    with open(out_file, 'wb') as fw:
        pickle.dump(ref_stat, fw)
        pickle.dump(target_stat, fw)
        pickle.dump(boundary, fw)
    return 1.0


def match_clusters_nearest_partial_samples(arguments):
    """
    do cluster matching and return match rate
    using minimum distance as distance value
    arguments: ref_clusters, candidate_node, candidate_label
    ref_clusters:
    candidate_node: node file contains node clusters
    candidate_label: retrieved label
    :return:
    """
    ref_node, candidate_node, num_class, out_dir = arguments

    # load reference clusters
    cls2ref_stat = {}
    with open(ref_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2ref_stat:
                    cls2ref_stat[int(label)] = statistic
            except EOFError:
                break

    # load dumped points
    ref_dump_file = os.path.join(
        '/'.join(ref_node.split('/')[:-2]),
        'ref_dumps',
        os.path.basename(ref_node)
    )
    all_points = []
    all_labels = []
    all_ids = []
    cls2cnt = {}
    with open(ref_dump_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                label = int(sp[2])
                if label not in cls2cnt:
                    cls2cnt[label] = 1
                else:
                    cls2cnt[label] += 1
                all_points.append(sp[1])
                all_labels.append(label)
                all_ids.append(sp[0])
            except EOFError:
                break
    kernel_dim = all_points[0].size
    all_points = np.array(all_points)

    # load target clusters
    cls2tgt_clt = {}
    cls2tgt_stat = {}
    with open(candidate_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2tgt_clt:
                    cls2tgt_clt[int(label)] = []
                cls2tgt_clt[int(label)].append(cluster)
                if int(label) not in cls2tgt_stat:
                    cls2tgt_stat[int(label)] = statistic
            except EOFError:
                break

    # key: ref_class-target_class, value: matched sample-id-set
    matched_samples = {}
    for rcli in cls2ref_stat.keys():
        ref_stat = cls2ref_stat[rcli]
        ref_mean, ref_std, ref_dim_order = ref_stat
        for tcli in cls2tgt_clt.keys():
            act_values = []
            for tci in cls2tgt_clt[tcli]:
                act_values.append(tci['max_act'])
            act_values = np.array(act_values)
            normed_values = act_values - np.min(act_values)
            total_weight = np.sum(np.exp(normed_values))
            tgt_mean, tgt_std, tgt_dim_order = cls2tgt_stat[tcli]
            trans_dim_order = np.zeros([kernel_dim], dtype=np.int32)
            for i in range(kernel_dim):
                ref_order = ref_dim_order[i]
                tgt_order = tgt_dim_order[i]
                trans_dim_order[ref_order] = tgt_order
            trans_dim_scale = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                scale = ref_std[ref_dim] + 1e-10
                trans_dim_scale[tgt_dim] = scale
            trans_dim_shift = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                trans_dim_shift[tgt_dim] = ref_mean[ref_dim]
            matched_dists = np.zeros([all_points.shape[0]], dtype=np.float32)
            matched_weight = np.zeros([all_points.shape[0]], dtype=np.float32)
            for idc, tci in enumerate(cls2tgt_clt[tcli]):
                target_center = tci['center']
                trans_center = (target_center * trans_dim_scale + trans_dim_shift)[trans_dim_order]
                dists = np.sqrt(np.sum((trans_center - all_points) ** 2, axis=1))
                weight = float(np.exp(tci['max_act'] - np.min(act_values)) / total_weight)
                for idx in range(dists.size):
                    if idc == 0:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
                    elif matched_dists[idx] > dists[idx]:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
            matched_dists = matched_dists / matched_weight
            sorted_idx = np.argsort(matched_dists)
            cls_avg_dist = {}
            points_class_count = {}
            max_idx = None
            ref_portion = None
            match_min_dist = None
            for sidx in range(sorted_idx.size):
                idx = sorted_idx[sidx]
                label = all_labels[idx]
                dist = matched_dists[idx]
                if label not in points_class_count:
                    points_class_count[label] = 1
                else:
                    points_class_count[label] += 1
                if label not in cls_avg_dist:
                    cls_avg_dist[label] = dist
                else:
                    cls_avg_dist[label] = ((points_class_count[label] - 1) / points_class_count[label]) \
                                          * cls_avg_dist[label] + dist * (1 / points_class_count[label])
                if sidx % int(sorted_idx.size / 20) == 0:
                    min_dist = None
                    matched_class = None
                    for label in cls_avg_dist.keys():
                        if min_dist is None or min_dist > cls_avg_dist[label]:
                            min_dist = cls_avg_dist[label]
                            matched_class = label
                    if matched_class == rcli:
                        max_idx = sidx
                        ref_portion = points_class_count[matched_class] / cls2cnt[matched_class]
                        match_min_dist = min_dist
            if max_idx is not None and ref_portion >= 0.3:
                samples = set()
                for sidx in range(max_idx + 1):
                    idx = sorted_idx[sidx]
                    samples.add(idx)
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(rcli)
                matched_samples[class_key] = [match_min_dist, samples]
            else:
                min_dist = None
                matched_class = None
                for label in cls_avg_dist.keys():
                    if min_dist is None or min_dist > cls_avg_dist[label]:
                        min_dist = cls_avg_dist[label]
                        matched_class = label
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(matched_class)
                matched_samples[class_key] = [min_dist, None]

    # build matched-dict and dist-dict
    match_ref_dic = {}
    for class_key in matched_samples.keys():
        ref_class, target_class, matched_class = class_key.split('-')
        ref_class = int(ref_class)
        target_class = int(target_class)
        matched_class = int(matched_class)
        transferred_dist = matched_samples[class_key][0]
        samples = matched_samples[class_key][1]
        if matched_class not in match_ref_dic:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]
        elif transferred_dist < match_ref_dic[matched_class][2]:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]

    matched_dict = {}
    dist_dict = {}
    for matched_class in match_ref_dic.keys():
        ref_class, target_class, transferred_dist, class_key, matched_sample_set = match_ref_dic[matched_class]
        if matched_sample_set is None:
            matched_sample_set = set()
            for sp_id in all_ids:
                if all_labels[sp_id] == matched_class:
                    matched_sample_set.add(sp_id)

        # dump information
        input_upper = None
        input_lower = None
        for idl, sp_id in enumerate(all_ids):
            if sp_id not in matched_sample_set:
                continue
            point = all_points[idl, :]
            if input_upper is None:
                input_upper = point
            else:
                input_upper = np.maximum(point, input_upper)
            if input_lower is None:
                input_lower = point
            else:
                input_lower = np.minimum(point, input_lower)

        target_sid = candidate_node.split('/')[-3]
        target_branch_node = os.path.basename(candidate_node).split('.')[0]
        if out_dir is None:
            ref_sid = ref_node.split('/')[-3]
        else:
            ref_sid = '_1'
        ref_branch_node = os.path.basename(ref_node).split('.')[0]
        target_key = '-'.join([target_sid, target_branch_node, str(target_class)])
        ref_key = '-'.join([ref_sid, ref_branch_node, str(matched_class)])
        full_match_key = ref_key + '+' + target_key
        out_path = out_dir
        if out_dir is None:
            out_path = os.path.dirname(ref_node)
        out_file = os.path.join(
            out_path,
            full_match_key + '.pkl'
        )
        if os.path.exists(out_file):
            print('\tfile exists:', out_file)
        dump_match_information(
            ref_stat=cls2ref_stat[ref_class],
            target_stat=cls2tgt_stat[target_class],
            out_file=out_file,
            boundary=[input_lower, input_upper],
        )
        if matched_class not in matched_dict:
            matched_dict[matched_class] = [target_class]
        else:
            matched_dict[matched_class].append(target_class)
        dist_key = str(matched_class) + '+' + str(target_class)
        dist_dict[dist_key] = transferred_dist

    return matched_dict, dist_dict, candidate_node


def match_clusters_stat_partial_samples(arguments):
    """
    do cluster matching and return match rate
    arguments: ref_clusters, candidate_node, candidate_label
    ref_clusters:
    candidate_node: node file contains node clusters
    candidate_label: retrieved label
    :return:
    """
    ref_node, candidate_node, num_class, out_dir, sample_rate, sample_weight = arguments

    # load dumped points
    ref_dump_file = os.path.join(
        '/'.join(ref_node.split('/')[:-2]),
        'ref_dumps',
        os.path.basename(ref_node)
    )
    cls2cnt = {}
    class_points = {}
    class_ids = {}
    with open(ref_dump_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                label = int(sp[2])
                if label not in cls2cnt:
                    cls2cnt[label] = 1
                else:
                    cls2cnt[label] += 1
                if label not in class_points:
                    class_points[label] = [sp[1]]
                else:
                    class_points[label].append(sp[1])
                if label not in class_ids:
                    class_ids[label] = [sp[0]]
                else:
                    class_ids[label].append(sp[0])
            except EOFError:
                break
    all_points = []
    all_labels = []
    all_ids = []
    # do sub-sample
    for cls in class_ids.keys():
        if sample_rate < 1.0:
            temp_ids = list(range(len(class_ids[cls])))
            to_sample = int(len(class_ids[cls]) * sample_rate)
            sub_ids = random.sample(temp_ids, to_sample)
            for sid in sub_ids:
                all_points.append(class_points[cls][sid])
                all_ids.append(class_ids[cls][sid])
                all_labels.append(cls)
        else:
            all_ids += class_ids[cls]
            all_points += class_points[cls]
            for i in range(len(class_ids[cls])):
                all_labels.append(cls)

    kernel_dim = all_points[0].size
    all_points = np.array(all_points)
    # get statistics by samples
    cls2ref_stat = {}
    for cls in cls2cnt.keys():
        stat_class_points = []
        sorted_dim_orders = np.zeros(shape=[kernel_dim, kernel_dim], dtype=np.float32)
        for i in range(len(all_labels)):
            if all_labels[i] == cls:
                stat_class_points.append(all_points[i])
                sorted_arg = np.argsort(all_points[i])
                for j in range(kernel_dim):
                    sorted_dim_orders[j, sorted_arg[j]] += 1
        stat_class_points = np.array(stat_class_points)
        class_mean = np.mean(stat_class_points, axis=0)
        upper_bound = np.max(stat_class_points, axis=0)
        lower_bound = np.min(stat_class_points, axis=0)
        dim_range = upper_bound - lower_bound
        sorted_dims = []
        dim_set = set()
        for j in range(kernel_dim):
            dims = np.argsort(sorted_dim_orders[j])
            for i in range(dims.size - 1, -1, -1):
                if dims[i] not in dim_set:
                    dim_set.add(dims[i])
                    sorted_dims.append(dims[i])
                    break
        sorted_dims = np.array(sorted_dims)
        cls2ref_stat[cls] = [class_mean, dim_range, sorted_dims]

    # load target clusters
    cls2tgt_clt = {}
    cls2tgt_stat = {}
    with open(candidate_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2tgt_clt:
                    cls2tgt_clt[int(label)] = []
                cls2tgt_clt[int(label)].append(cluster)
                if int(label) not in cls2tgt_stat:
                    cls2tgt_stat[int(label)] = statistic
            except EOFError:
                break

    # key: ref_class-target_class, value: matched sample-id-set
    matched_samples = {}
    for rcli in cls2ref_stat.keys():
        ref_stat = cls2ref_stat[rcli]
        ref_mean, ref_std, ref_dim_order = ref_stat
        for tcli in cls2tgt_clt.keys():
            act_values = []
            for tci in cls2tgt_clt[tcli]:
                act_values.append(tci['max_act'])
            act_values = np.array(act_values)
            normed_values = act_values - np.min(act_values)
            total_weight = np.sum(np.exp(normed_values))
            trans_dim_order = np.zeros([kernel_dim], dtype=np.int32)
            for i in range(kernel_dim):
                ref_order = ref_dim_order[i]
                tgt_order = i
                trans_dim_order[ref_order] = tgt_order
            trans_dim_scale = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                scale = ref_std[ref_dim] + 1e-10
                trans_dim_scale[tgt_dim] = scale
            trans_dim_shift = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                trans_dim_shift[tgt_dim] = ref_mean[ref_dim]
            matched_dists = np.zeros([all_points.shape[0]], dtype=np.float32)
            matched_weight = np.zeros([all_points.shape[0]], dtype=np.float32)
            for idc, tci in enumerate(cls2tgt_clt[tcli]):
                target_center = tci['center']
                trans_center = (target_center * trans_dim_scale + trans_dim_shift)[trans_dim_order]
                dists = np.sqrt(np.sum((trans_center - all_points) ** 2, axis=1))
                weight = float(np.exp(tci['max_act'] - np.min(act_values)) / total_weight)
                for idx in range(dists.size):
                    if idc == 0:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
                    elif matched_dists[idx] > dists[idx]:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
            matched_dists = matched_dists / matched_weight
            sorted_idx = np.argsort(matched_dists)
            cls_avg_dist = {}
            points_class_count = {}
            max_idx = None
            ref_portion = None
            match_min_dist = None
            for sidx in range(sorted_idx.size):
                idx = sorted_idx[sidx]
                label = all_labels[idx]
                dist = matched_dists[idx]
                if label not in points_class_count:
                    points_class_count[label] = 1
                else:
                    points_class_count[label] += 1
                if label not in cls_avg_dist:
                    cls_avg_dist[label] = dist
                else:
                    cls_avg_dist[label] = ((points_class_count[label] - 1) / points_class_count[label]) \
                                          * cls_avg_dist[label] + dist * (1 / points_class_count[label])
                if sidx % int(sorted_idx.size / 20) == 0:
                    min_dist = None
                    matched_class = None
                    for label in cls_avg_dist.keys():
                        if min_dist is None or min_dist > cls_avg_dist[label]:
                            min_dist = cls_avg_dist[label]
                            matched_class = label
                    if matched_class == rcli:
                        max_idx = sidx
                        ref_portion = points_class_count[matched_class] / cls2cnt[matched_class]
                        match_min_dist = min_dist
            if max_idx is not None and ref_portion >= 0.3:
                samples = set()
                for sidx in range(max_idx + 1):
                    idx = sorted_idx[sidx]
                    samples.add(all_ids[idx])
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(rcli)
                matched_samples[class_key] = [match_min_dist, samples]
            else:
                min_dist = None
                matched_class = None
                for label in cls_avg_dist.keys():
                    if min_dist is None or min_dist > cls_avg_dist[label]:
                        min_dist = cls_avg_dist[label]
                        matched_class = label
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(matched_class)
                matched_samples[class_key] = [min_dist, None]

    # build matched-dict and dist-dict
    match_ref_dic = {}
    for class_key in matched_samples.keys():
        ref_class, target_class, matched_class = class_key.split('-')
        ref_class = int(ref_class)
        target_class = int(target_class)
        matched_class = int(matched_class)
        transferred_dist = matched_samples[class_key][0]
        samples = matched_samples[class_key][1]
        if matched_class not in match_ref_dic:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]
        elif transferred_dist < match_ref_dic[matched_class][2]:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]

    matched_dict = {}
    dist_dict = {}
    for matched_class in match_ref_dic.keys():
        ref_class, target_class, transferred_dist, class_key, matched_sample_set = match_ref_dic[matched_class]
        if matched_sample_set is None:
            matched_sample_set = set()
            for idx, sp_id in enumerate(all_ids):
                if all_labels[idx] == matched_class:
                    matched_sample_set.add(sp_id)

        # dump information
        input_upper = None
        input_lower = None
        for idl, sp_id in enumerate(all_ids):
            if sp_id not in matched_sample_set:
                continue
            point = all_points[idl, :]
            if input_upper is None:
                input_upper = point
            else:
                input_upper = np.maximum(point, input_upper)
            if input_lower is None:
                input_lower = point
            else:
                input_lower = np.minimum(point, input_lower)

        target_sid = candidate_node.split('/')[-3]
        target_branch_node = os.path.basename(candidate_node).split('.')[0]
        if out_dir is None:
            ref_sid = ref_node.split('/')[-3]
        else:
            ref_sid = '_1'
        ref_branch_node = os.path.basename(ref_node).split('.')[0]
        target_key = '-'.join([target_sid, target_branch_node, str(target_class)])
        ref_key = '-'.join([ref_sid, ref_branch_node, str(matched_class)])
        full_match_key = ref_key + '+' + target_key
        out_path = out_dir
        if out_dir is None:
            out_path = os.path.dirname(ref_node)
        out_file = os.path.join(
            out_path,
            full_match_key + '.pkl'
        )
        if os.path.exists(out_file):
            print('\tfile exists:', out_file)
        dump_match_information(
            ref_stat=cls2ref_stat[ref_class],
            target_stat=cls2tgt_stat[target_class],
            out_file=out_file,
            boundary=[input_lower, input_upper],
        )
        if matched_class not in matched_dict:
            matched_dict[matched_class] = [target_class]
        else:
            matched_dict[matched_class].append(target_class)
        dist_key = str(matched_class) + '+' + str(target_class)
        dist_dict[dist_key] = transferred_dist

    return matched_dict, dist_dict, candidate_node


def match_clusters_weighted_partial_samples(arguments):
    """
    do cluster matching and return match rate
    arguments: ref_clusters, candidate_node, candidate_label
    ref_clusters:
    candidate_node: node file contains node clusters
    candidate_label: retrieved label
    :return:
    """
    ref_node, candidate_node, num_class, out_dir, sample_rate, sample_weight = arguments

    # load dumped points
    if out_dir is None:
        ref_dump_file = os.path.join(
            '/'.join(ref_node.split('/')[:-2]),
            'ref_dumps',
            os.path.basename(ref_node)
        )
    else:
        ref_dump_file = ref_node
    cls2cnt = {}
    all_points = []
    all_labels = []
    all_ids = []
    with open(ref_dump_file, 'rb') as fr:
        while True:
            try:
                sp = pickle.load(fr)
                label = int(sp[2])
                if label not in cls2cnt:
                    cls2cnt[label] = 1
                else:
                    cls2cnt[label] += 1
                all_ids.append(int(sp[0]))
                all_points.append(sp[1])
                all_labels.append(label)
            except EOFError:
                break

    kernel_dim = all_points[0].size
    all_points = np.array(all_points)
    # get statistics by samples
    cls2ref_stat = {}
    for cls in cls2cnt.keys():
        class_points = []
        class_mean = np.zeros([kernel_dim], dtype=np.float32)
        sorted_dim_orders = np.zeros(shape=[kernel_dim, kernel_dim], dtype=np.float32)
        for i in range(len(all_labels)):
            if all_labels[i] == cls:
                sp_id = all_ids[i]
                class_mean += all_points[i] * sample_weight[cls][sp_id]
                class_points.append(all_points[i])
                sorted_arg = np.argsort(all_points[i])
                for j in range(kernel_dim):
                    sorted_dim_orders[j, sorted_arg[j]] += sample_weight[cls][sp_id]
        class_points = np.array(class_points)
        upper_bound = np.max(class_points, axis=0)
        lower_bound = np.min(class_points, axis=0)
        dim_range = upper_bound - lower_bound
        sorted_dims = []
        dim_set = set()
        for j in range(kernel_dim):
            dims = np.argsort(sorted_dim_orders[j])
            for i in range(dims.size - 1, -1, -1):
                if dims[i] not in dim_set:
                    dim_set.add(dims[i])
                    sorted_dims.append(dims[i])
                    break
        sorted_dims = np.array(sorted_dims)
        cls2ref_stat[cls] = [class_mean, dim_range, sorted_dims]

    # load target clusters
    cls2tgt_clt = {}
    cls2tgt_stat = {}
    with open(candidate_node, 'rb') as fr:
        while True:
            try:
                cluster_info = pickle.load(fr)
                cluster, statistic, label = cluster_info
                if int(label) not in cls2tgt_clt:
                    cls2tgt_clt[int(label)] = []
                cls2tgt_clt[int(label)].append(cluster)
                if int(label) not in cls2tgt_stat:
                    cls2tgt_stat[int(label)] = statistic
            except EOFError:
                break

    # key: ref_class-target_class, value: matched sample-id-set
    matched_samples = {}
    for rcli in cls2ref_stat.keys():
        ref_stat = cls2ref_stat[rcli]
        ref_mean, ref_std, ref_dim_order = ref_stat
        for tcli in cls2tgt_clt.keys():
            act_values = []
            for tci in cls2tgt_clt[tcli]:
                act_values.append(tci['max_act'])
            act_values = np.array(act_values)
            normed_values = act_values - np.min(act_values)
            total_weight = np.sum(np.exp(normed_values))
            trans_dim_order = np.zeros([kernel_dim], dtype=np.int32)
            for i in range(kernel_dim):
                ref_order = ref_dim_order[i]
                tgt_order = i
                trans_dim_order[ref_order] = tgt_order
            trans_dim_scale = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                scale = ref_std[ref_dim] + 1e-10
                trans_dim_scale[tgt_dim] = scale
            trans_dim_shift = np.zeros([kernel_dim], dtype=np.float32)
            for i in range(kernel_dim):
                ref_dim = i
                tgt_dim = trans_dim_order[i]
                trans_dim_shift[tgt_dim] = ref_mean[ref_dim]
            matched_dists = np.zeros([all_points.shape[0]], dtype=np.float32)
            matched_weight = np.zeros([all_points.shape[0]], dtype=np.float32)
            for idc, tci in enumerate(cls2tgt_clt[tcli]):
                target_center = tci['center']
                trans_center = (target_center * trans_dim_scale + trans_dim_shift)[trans_dim_order]
                dists = np.sqrt(np.sum((trans_center - all_points) ** 2, axis=1))
                weight = float(np.exp(tci['max_act'] - np.min(act_values)) / total_weight)
                for idx in range(dists.size):
                    if idc == 0:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
                    elif matched_dists[idx] > dists[idx]:
                        matched_dists[idx] = dists[idx]
                        matched_weight[idx] = weight
            matched_dists = matched_dists / matched_weight
            sorted_idx = np.argsort(matched_dists)
            cls_avg_dist = {}
            points_class_count = {}
            max_idx = None
            ref_portion = None
            match_min_dist = None
            for sidx in range(sorted_idx.size):
                idx = sorted_idx[sidx]
                label = all_labels[idx]
                dist = matched_dists[idx]
                if label not in points_class_count:
                    points_class_count[label] = 1
                else:
                    points_class_count[label] += 1
                if label not in cls_avg_dist:
                    cls_avg_dist[label] = dist
                else:
                    cls_avg_dist[label] = ((points_class_count[label] - 1) / points_class_count[label]) \
                                          * cls_avg_dist[label] + dist * (1 / points_class_count[label])
                if sidx % int(sorted_idx.size / 20) == 0:
                    min_dist = None
                    matched_class = None
                    for label in cls_avg_dist.keys():
                        if min_dist is None or min_dist > cls_avg_dist[label]:
                            min_dist = cls_avg_dist[label]
                            matched_class = label
                    if matched_class == rcli:
                        max_idx = sidx
                        ref_portion = points_class_count[matched_class] / cls2cnt[matched_class]
                        match_min_dist = min_dist
            if max_idx is not None and ref_portion >= 0.3:
                samples = set()
                for sidx in range(max_idx + 1):
                    idx = sorted_idx[sidx]
                    samples.add(all_ids[idx])
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(rcli)
                matched_samples[class_key] = [match_min_dist, samples]
            else:
                min_dist = None
                matched_class = None
                for label in cls_avg_dist.keys():
                    if min_dist is None or min_dist > cls_avg_dist[label]:
                        min_dist = cls_avg_dist[label]
                        matched_class = label
                class_key = str(rcli) + '-' + str(tcli) + '-' + str(matched_class)
                matched_samples[class_key] = [min_dist, None]

    # build matched-dict and dist-dict
    match_ref_dic = {}
    for class_key in matched_samples.keys():
        ref_class, target_class, matched_class = class_key.split('-')
        ref_class = int(ref_class)
        target_class = int(target_class)
        matched_class = int(matched_class)
        transferred_dist = matched_samples[class_key][0]
        samples = matched_samples[class_key][1]
        if matched_class not in match_ref_dic:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]
        elif transferred_dist < match_ref_dic[matched_class][2]:
            match_ref_dic[matched_class] = [ref_class, target_class, transferred_dist, class_key, samples]

    matched_dict = {}
    dist_dict = {}
    for matched_class in match_ref_dic.keys():
        ref_class, target_class, transferred_dist, class_key, matched_sample_set = match_ref_dic[matched_class]
        if matched_sample_set is None:
            matched_sample_set = set()
            for idx, sp_id in enumerate(all_ids):
                if all_labels[idx] == matched_class:
                    matched_sample_set.add(sp_id)

        # dump information
        input_upper = None
        input_lower = None
        for idl, sp_id in enumerate(all_ids):
            if sp_id not in matched_sample_set:
                continue
            point = all_points[idl, :]
            if input_upper is None:
                input_upper = point
            else:
                input_upper = np.maximum(point, input_upper)
            if input_lower is None:
                input_lower = point
            else:
                input_lower = np.minimum(point, input_lower)

        target_sid = candidate_node.split('/')[-3]
        target_branch_node = os.path.basename(candidate_node).split('.')[0]
        if out_dir is None:
            ref_sid = ref_node.split('/')[-3]
        else:
            ref_sid = '_1'
        ref_branch_node = os.path.basename(ref_node).split('.')[0]
        target_key = '-'.join([target_sid, target_branch_node, str(target_class)])
        ref_key = '-'.join([ref_sid, ref_branch_node, str(matched_class)])
        full_match_key = ref_key + '+' + target_key
        out_path = out_dir
        if out_dir is None:
            out_path = os.path.dirname(ref_node)
        out_file = os.path.join(
            out_path,
            full_match_key + '.pkl'
        )
        if os.path.exists(out_file):
            print('\tfile exists:', out_file)
        dump_match_information(
            ref_stat=cls2ref_stat[ref_class],
            target_stat=cls2tgt_stat[target_class],
            out_file=out_file,
            boundary=[input_lower, input_upper],
        )
        if matched_class not in matched_dict:
            matched_dict[matched_class] = [target_class]
        else:
            matched_dict[matched_class].append(target_class)
        dist_key = str(matched_class) + '+' + str(target_class)
        dist_dict[dist_key] = transferred_dist

    return matched_dict, dist_dict, candidate_node


def parallel_class_match(ref_file,
                         target_files,
                         threads,
                         num_class,
                         match_data=False,
                         dump_path=None,
                         sample_rate=1.0,
                         sample_weight=None):
    """
    match and find the closest class-cluster of reference node
    :param ref_file:
    :param target_files:
    :param threads:
    :param num_class:
    :param match_data:
    :param dump_path:
    :param sample_rate:
    :param sample_weight:
    :return:
    """
    arg_list = []
    for ti in target_files:
        if match_data:
            node = os.path.basename(ref_file).split('.')[0]
            out_dir = os.path.join(dump_path, node)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            arg_list.append([ref_file, ti, num_class, out_dir, sample_rate, sample_weight])
        else:
            arg_list.append([ref_file, ti, num_class, None, sample_rate, sample_weight])

    # parallel match
    pool = Pool(processes=threads)
    match_results = pool.map(match_clusters_weighted_partial_samples, arg_list)
    pool.close()
    pool.join()
    gc.collect()

    # key: reference_sid-reference_node-reference_class
    # value: set(target_sid-target_node-target_class)
    if match_data:
        ref_sid = '_1'
    else:
        ref_sid = ref_file.split('/')[-3]
    ref_node = ref_file.split('/')[-1].split('.')[0]
    total_matched_dict = {}
    total_dist_dict = {}
    for mri in match_results:
        matched_dic, dist_dict, target_file = mri
        target_sid = target_file.split('/')[-3]
        target_node = target_file.split('/')[-1].split('.')[0]
        for cli in matched_dic.keys():
            ref_key = '-'.join([ref_sid, ref_node, str(cli)])
            if ref_key not in total_matched_dict:
                total_matched_dict[ref_key] = set()
            for tcli in matched_dic[cli]:
                tgt_key = '-'.join([target_sid, target_node, str(tcli)])
                total_matched_dict[ref_key].add(tgt_key)
                dist_key = ref_key + '+' + tgt_key
                total_dist_dict[dist_key] = dist_dict[str(cli) + '+' + str(tcli)]
    return total_matched_dict, total_dist_dict
