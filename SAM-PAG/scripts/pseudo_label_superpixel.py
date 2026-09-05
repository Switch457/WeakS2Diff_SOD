import cv2  # type: ignore
from PIL import Image
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


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
        #combined_segmentation = morphology.remove_small_objects(combined_segmentation, min_size=50)

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



def label_from_superpixel(color_image_ori, depth_image_ori, gt_ori, mask_ori, superpixel, sam, base):

        
        gt = Image.fromarray(gt_ori, mode = 'L')
        mask = Image.fromarray(mask_ori, mode = 'L')
        color_image = np.array(color_image_ori)
        depth_image = np.array(depth_image_ori)
        #pseudo_label = Image.fromarray(pseudo_label)

        color_masks = []
        depth_masks = []

        # 处理彩色图
        color_predictor = SamPredictor(sam)
        color_predictor.set_image(color_image)
        #predictor.set_image(color_image)

        # 处理深度图
        depth_predictor = SamPredictor(sam)
        depth_predictor.set_image(depth_image)
        depth_predictor.set_image(depth_image)

        for i in range(5):

        
            input_point_superpixel, input_label_superpixel = get_superpixel_point(superpixel, 10, superpixel.size[0], superpixel.size[1])
            input_point_scribble, input_label_scribble = get_prompt(gt, mask, 10, gt.size[0], gt.size[1])
    

            input_point = torch.cat((input_point_superpixel, input_point_scribble), dim = 0)
            input_label = torch.cat((input_label_superpixel, input_label_scribble), dim = 0)


            # 转换点和标签到合适的设备
            input_point = input_point.to(device="cuda")   
            input_label = input_label.to(device="cuda")    

            # 彩色图掩码生成 - 使用logits和sigmoid处理
            color_logits, color_scores, _ = color_predictor.predict(
                point_coords=input_point.cpu().numpy(),
                point_labels=input_label.cpu().numpy(),
                multimask_output=False,
                return_logits=True
            )


            # 深度图掩码生成 - 使用logits和sigmoid处理
            depth_logits, depth_scores, _ = depth_predictor.predict(
                point_coords=input_point.cpu().numpy(),
                point_labels=input_label.cpu().numpy(),
                multimask_output=False,
                return_logits=True
            )

            color_probs = torch.sigmoid(torch.tensor(color_logits[0]))  # 使用sigmoid转换为概率
            color_single_mask = (color_probs > 0.5).float().numpy()  # 阈值化为二值掩码

            depth_probs = torch.sigmoid(torch.tensor(depth_logits[0]))  # 使用sigmoid转换为概率
            depth_single_mask = (depth_probs > 0.5).float().numpy()  # 阈值化为二值掩码


            # 手动添加面积和边界框信息
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
            
            color_masks.append({
                "segmentation": color_single_mask,
                "area": color_area,
                "bbox": [color_x0, color_y0, color_w, color_h],
                "point_coords": input_point.cpu().numpy(),
                "point_labels":input_label.cpu().numpy(),
                "stability_score": color_scores[0],
                "crop_box": [0, 0, color_image.shape[1], color_image.shape[0]],
                "probabilities": color_probs.numpy()  # 保存概率图
            })

            depth_masks.append({
                "segmentation": depth_single_mask,
                "area": depth_area,
                "bbox": [depth_x0, depth_y0, depth_w, depth_h],
                "point_coords": input_point.cpu().numpy(),##"point_coords": [input_point_depth.cpu().numpy()[0]],
                "point_labels":input_label.cpu().numpy(),
                "stability_score": depth_scores[0],
                #"crop_box": [0, 0, depth_image_3ch.shape[1], depth_image_3ch.shape[0]],
                "crop_box": [0, 0, depth_image.shape[1], depth_image.shape[0]],
                "probabilities": depth_probs.numpy()  # 保存概率图
            })

            
        combined_masks = weighted_fusion(color_masks, depth_masks)
        

        return combined_masks
      