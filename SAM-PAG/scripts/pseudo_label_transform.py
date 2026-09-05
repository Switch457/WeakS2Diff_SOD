import cv2  # type: ignore
from PIL import Image
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os
from typing import Any, Dict, List
from transform import cv_flip, cv_resize, cv_rotation, flip_points, resize_points, rotation_points

def get_prompt(x, y, points_num, width, height):
    point_list = []
    label_list = []
    t = 0
    while len(point_list) < points_num and t < 10000:
        t = t + 1
        random_x = np.random.randint(0, width)
        random_y = np.random.randint(0, height)
        x_value = x.getpixel((random_x, random_y))
        y_value = y.getpixel((random_x, random_y))
        # 判断前景或者背景
        if x_value == 255 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(1)
        elif x_value == 0 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(0)

    if len(point_list) < 1:
        point_list.append([0, 0])
        label_list.append(-1)

    # 将随机选择的点坐标转换为tensor张量作为输入点
    input_point = torch.tensor(point_list)
    # 定义对应输入点的标签
    input_label = torch.tensor(label_list)
    return input_point, input_label


def get_superpixel_point(mask, points_num, width, height):
    point_list = []
    label_list = []
    t = 0
    while len(point_list) < points_num and t < 10000:
        t = t + 1
        random_x = np.random.randint(0, width)
        random_y = np.random.randint(0, height)
        value = mask.getpixel((random_x, random_y))
        # 判断前景或者背景
        if value == 255:
            point_list.append([random_x, random_y])
            label_list.append(1)
        elif value == 0:
            if label_list.count(0) > label_list.count(1):
                continue
            point_list.append([random_x, random_y])
            label_list.append(0)

    # 将随机选择的点坐标转换为tensor张量作为输入点
    input_point = torch.tensor(point_list)
    # 定义对应输入点的标签
    input_label = torch.tensor(label_list)

    return input_point, input_label

def calculate_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8), (mask1.shape[1], mask1.shape[0]))
        mask2 = mask2.astype(np.bool_)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou

def weighted_fusion(color_masks, depth_masks, color_weight=0.5, depth_weight=0.5):
    combined_masks = []
    for color_mask, depth_mask in zip(color_masks, depth_masks):
        # 基于预测置信度调整权重
        color_score = color_mask["stability_score"]
        depth_score = depth_mask["stability_score"]
        total_score = color_score + depth_score
        if total_score > 0:
            color_weight = color_score / total_score
            depth_weight = depth_score / total_score

        # 使用概率值进行加权融合
        color_prob = color_mask["probabilities"]
        depth_prob = depth_mask["probabilities"]
        combined_prob = color_weight * color_prob + depth_weight * depth_prob
        combined_segmentation = (combined_prob > 0.5).astype(np.uint8)

        # 后处理：形态学操作，优化参数
        #combined_segmentation = morphology.remove_small_holes(combined_segmentation, area_threshold=50)
        #ombined_segmentation = morphology.remove_small_objects(combined_segmentation, min_size=50)

        area = np.count_nonzero(combined_segmentation)
        rows, cols = np.where(combined_segmentation)
        if len(rows) > 0 and len(cols) > 0:
            x0 = np.min(cols)
            y0 = np.min(rows)
            w = np.max(cols) - x0 + 1
            h = np.max(rows) - y0 + 1
        else:
            x0, y0, w, h = 0, 0, 0, 0

        combined_masks.append({
            "segmentation": combined_segmentation,
            "area": area,
            "bbox": [x0, y0, w, h],
            "point_coords": color_mask["point_coords"],
            "point_labels":color_mask["point_labels"],
            "stability_score": (color_mask["stability_score"] + depth_mask["stability_score"]) / 2,
            "crop_box": color_mask["crop_box"],
            "probabilities": combined_prob,  # 保存融合后的概率图
            "IoU":calculate_iou(color_mask["segmentation"], depth_mask["segmentation"]) # 深度与RGB预测图的IoU
        })
    return combined_masks

 
def label_from_transform(color_image_ori, depth_image_ori, gt_ori, mask_ori, superpixel_ori, sam, base):

    color_masks_transform = []
    depth_masks_transform = []

    gt = Image.fromarray(gt_ori)
    mask = Image.fromarray(mask_ori)
    width, height = gt.size

    # 1. 先固定 5 组 prompt，保证所有变换版本使用相同的 prompt sets
    prompt_sets = []
    for i in range(5):
        input_point_ori, input_label = get_prompt(gt, mask, 20, width, height)
        prompt_sets.append((input_point_ori, input_label))

    num_prompt_sets = len(prompt_sets)

    color_logits_fusions = [None for _ in range(num_prompt_sets)]
    depth_logits_fusions = [None for _ in range(num_prompt_sets)]
    color_scores_fusions = [0 for _ in range(num_prompt_sets)]
    depth_scores_fusions = [0 for _ in range(num_prompt_sets)]

    points_records = [None for _ in range(num_prompt_sets)]
    labels_records = [None for _ in range(num_prompt_sets)]

    # 复用 predictor；每次 set_image 会覆盖当前缓存的 image embedding
    color_predictor = SamPredictor(sam)
    depth_predictor = SamPredictor(sam)

    last_color_image = color_image_ori
    last_depth_image = depth_image_ori

    # 2. 先遍历 4 个图像版本，每个版本只 set_image 一次
    for j in range(4):

        if j == 1:
            color_image = cv_flip(color_image_ori)
            depth_image = cv_flip(depth_image_ori)
        elif j == 2:
            color_image = cv_resize(color_image_ori, gt_ori.shape[1] // 2, gt_ori.shape[0] // 2)
            depth_image = cv_resize(depth_image_ori, gt_ori.shape[1] // 2, gt_ori.shape[0] // 2)
        elif j == 3:
            color_image = cv_rotation(color_image_ori, 1)
            depth_image = cv_rotation(depth_image_ori, 1)
        else:
            color_image = color_image_ori
            depth_image = depth_image_ori

        last_color_image = color_image
        last_depth_image = depth_image

        # 当前变换版本只计算一次 image embedding
        color_predictor.set_image(color_image)
        depth_predictor.set_image(depth_image)

        # 该 image embedding 复用给 5 组 prompt
        for i, (input_point_ori, input_label) in enumerate(prompt_sets):

            if j == 1:
                input_point = flip_points(input_point_ori, width)
            elif j == 2:
                input_point = resize_points(input_point_ori)
            elif j == 3:
                input_point = rotation_points(input_point_ori, width)
            else:
                input_point = input_point_ori

            input_point = input_point.to(device="cuda")
            input_label = input_label.to(device="cuda")

            if j == 0:
                points_records[i] = input_point
                labels_records[i] = input_label

            color_logits, color_scores, _ = color_predictor.predict(
                point_coords=input_point.cpu().numpy(),
                point_labels=input_label.cpu().numpy(),
                multimask_output=False,
                return_logits=True
            )

            depth_logits, depth_scores, _ = depth_predictor.predict(
                point_coords=input_point.cpu().numpy(),
                point_labels=input_label.cpu().numpy(),
                multimask_output=False,
                return_logits=True
            )

            # 逆变换回原图空间
            if j == 1:
                color_logits_inv = cv_flip(color_logits[0])
                depth_logits_inv = cv_flip(depth_logits[0])
            elif j == 2:
                color_logits_inv = cv_resize(color_logits[0], gt_ori.shape[1], gt_ori.shape[0])
                depth_logits_inv = cv_resize(depth_logits[0], gt_ori.shape[1], gt_ori.shape[0])
            elif j == 3:
                color_logits_inv = cv_rotation(color_logits[0], -1).copy()
                depth_logits_inv = cv_rotation(depth_logits[0], -1).copy()
            else:
                color_logits_inv = color_logits[0]
                depth_logits_inv = depth_logits[0]

            color_logits_inv = np.asarray(color_logits_inv, dtype=np.float32)
            depth_logits_inv = np.asarray(depth_logits_inv, dtype=np.float32)

            if color_logits_fusions[i] is None:
                color_logits_fusions[i] = color_logits_inv
                depth_logits_fusions[i] = depth_logits_inv
            else:
                color_logits_fusions[i] += color_logits_inv
                depth_logits_fusions[i] += depth_logits_inv

            color_scores_fusions[i] += color_scores
            depth_scores_fusions[i] += depth_scores

    # 3. 对每组 prompt 下的 4 个变换结果做平均融合
    for i in range(num_prompt_sets):
        color_logits_fusion = color_logits_fusions[i] / 4
        depth_logits_fusion = depth_logits_fusions[i] / 4
        color_scores_fusion = color_scores_fusions[i] / 4
        depth_scores_fusion = depth_scores_fusions[i] / 4

        color_probs = torch.sigmoid(torch.tensor(color_logits_fusion))
        color_single_mask = (color_probs > 0.5).float().numpy()

        depth_probs = torch.sigmoid(torch.tensor(depth_logits_fusion))
        depth_single_mask = (depth_probs > 0.5).float().numpy()

        color_area = np.count_nonzero(color_single_mask)
        color_rows, color_cols = np.where(color_single_mask)
        if len(color_rows) > 0 and len(color_cols) > 0:
            color_x0 = np.min(color_cols)
            color_y0 = np.min(color_rows)
            color_w = np.max(color_cols) - color_x0 + 1
            color_h = np.max(color_rows) - color_y0 + 1
        else:
            color_x0, color_y0, color_w, color_h = 0, 0, 0, 0

        depth_area = np.count_nonzero(depth_single_mask)
        depth_rows, depth_cols = np.where(depth_single_mask)
        if len(depth_rows) > 0 and len(depth_cols) > 0:
            depth_x0 = np.min(depth_cols)
            depth_y0 = np.min(depth_rows)
            depth_w = np.max(depth_cols) - depth_x0 + 1
            depth_h = np.max(depth_rows) - depth_y0 + 1
        else:
            depth_x0, depth_y0, depth_w, depth_h = 0, 0, 0, 0

        points = points_records[i]
        labels = labels_records[i]

        color_masks_transform.append({
            "segmentation": color_single_mask,
            "area": color_area,
            "bbox": [color_x0, color_y0, color_w, color_h],
            "point_coords": points.cpu().numpy(),
            "point_labels": labels.cpu().numpy(),
            "stability_score": color_scores_fusion[0],
            "crop_box": [0, 0, last_color_image.shape[1], last_color_image.shape[0]],
            "probabilities": color_probs.numpy()
        })

        depth_masks_transform.append({
            "segmentation": depth_single_mask,
            "area": depth_area,
            "bbox": [depth_x0, depth_y0, depth_w, depth_h],
            "point_coords": points.cpu().numpy(),
            "point_labels": labels.cpu().numpy(),
            "stability_score": depth_scores_fusion[0],
            "crop_box": [0, 0, last_depth_image.shape[1], last_depth_image.shape[0]],
            "probabilities": depth_probs.numpy()
        })

    combined_masks = weighted_fusion(color_masks_transform, depth_masks_transform)

    return combined_masks
