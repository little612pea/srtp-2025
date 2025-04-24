# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import mmengine
import numpy as np

from mmaction.apis import detection_inference, pose_inference
from mmaction.utils import frame_extract
from mmpose.apis import init_model
from mmdet.apis import init_detector
import os
os.environ['CUDA_HOME'] = "/home/jovyan/cuda-117"

args = abc.abstractproperty()
args.det_config = '/home/jovyan/2024-srtp/mmaction2/demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501
args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = '/home/jovyan/2024-srtp/mmaction2/demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501

args.pose_config = init_model(args.pose_config, args.pose_checkpoint, "cuda:0")
args.det_config = init_detector(args.det_config, args.det_checkpoint, "cuda:0")

def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, threshold=0.8):
        return intersection(box0, box1) / area(box0) > threshold

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, threshold=threshold):
        shape = [sum(bbox[:, -1] > threshold) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox[:, None, :]


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))


def pose_inference_with_align(args, frame_paths, det_results,vid):
    # filter frame without det bbox
    det_results = [
        frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0
    ]

    pose_results, _ = pose_inference(args.pose_config, args.pose_checkpoint,
                                     frame_paths, det_results, args.device)
    # align the num_person among frames
    if not pose_results or all(pose['keypoints'].shape[0] == 0 for pose in pose_results):
        print(f"Skipping video {osp.basename(vid)} due to no valid keypoints detected.")
        return None, None
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)
    keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                         dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    return keypoints, scores


def ntu_pose_extraction(vid, skip_postproc=False):
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = frame_extract(vid, out_dir=tmp_dir.name)
    det_results, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        device=args.device,
        with_score=True)

    if not skip_postproc:
        print(f'\nProcessing {vid}')
        det_results = ntu_det_postproc(vid, det_results)

    anno = dict()

    keypoints, scores = pose_inference_with_align(args, frame_paths,
                                                  det_results,vid)
    if keypoints is None or scores is None:
        print(f"Skipping video {osp.basename(vid)} due to no valid keypoints detected.")
        return None
    anno['keypoint'] = keypoints
    anno['keypoint_score'] = scores
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['total_frames'] = keypoints.shape[1]
    anno['label'] = int(osp.basename(vid).split('A')[1][:3]) - 1
    tmp_dir.cleanup()

    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('output', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    args = parser.parse_args()
    return args


# if __name__ == '__main__':
#     global_args = parse_args()
#     args.device = global_args.device
#     args.video = global_args.video
#     args.output = global_args.output
#     args.skip_postproc = global_args.skip_postproc
#     anno = ntu_pose_extraction(args.video, args.skip_postproc)
#     mmengine.dump(anno, args.output)

from pathlib import Path
def process_videos(input_folder, output_folder):
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
 # 使用os.walk递归遍历文件夹
    for root,dirs,files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().endswith('.mp4'):
                input_file_path = os.path.join(root, file_name)
                
                args.video = input_file_path  # 输入视频的完整路径
                
                # 创建对应的输出文件路径，保留原始文件夹结构
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                Path(output_subfolder).mkdir(parents=True, exist_ok=True)
                
                args.output = os.path.join(output_subfolder, file_name + ".pkl")  # 输出文件路径
                                # 如果输出文件已存在，则跳过
                if os.path.exists(args.output):
                    print(f"File already exists, skipping: {args.output}")
                    continue
                args.skip_postproc = True
                args.device = 'cuda:0'
                # 执行姿态提取
                anno = ntu_pose_extraction(args.video, args.skip_postproc)
                
                # 将注释信息保存到输出文件
                mmengine.dump(anno, args.output)
                print(f"Processed and saved: {args.output}")

# 使用方法
input_directory = '/home/jovyan/2024-srtp/mmaction2/k400'  # 替换为你的输入文件夹路径
output_directory = '/home/jovyan/2024-srtp/mmaction2/k400-processed'  # 替换为你的输出文件夹路径
# 假设global_args已经定义并且包含必要的属性如device和skip_postproc
process_videos(input_directory, output_directory)