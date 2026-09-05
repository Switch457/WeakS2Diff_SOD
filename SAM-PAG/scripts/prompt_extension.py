import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch
import cv2
from fast_slic import Slic
import argparse  #
import time

def process_superpixel(superpixel_map, gt, num_components=70):
    """
    Generate superpixel labels based on overlap with ground truth.
    """
    labels = []
    label_gt = np.zeros((1, gt.shape[0], gt.shape[1])) #(height,width)

    for i in range(1, num_components + 1):
        superpixel_mask = (superpixel_map == i).astype(np.float32)
        if np.sum(superpixel_mask) != 0:
            if np.sum(superpixel_mask * gt) > 1:
                label_gt += superpixel_mask
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)

    label_gt = torch.tensor(label_gt).to(torch.float32)
    return labels, label_gt

def prompt_extension(np_img, np_depth, np_gt):  
    slic = Slic(num_components=70, compactness=10)
    SS_map = slic.iterate(np_img) + 1
    SS_map_depth = slic.iterate(np_depth) + 1

    SS_maps_label, label_gt = process_superpixel(SS_map, np_gt)
    SS_maps_label_depth, label_gt_depth = process_superpixel(SS_map_depth, np_gt)
    return label_gt, label_gt_depth


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    default= "./train_data/img/",
    help="Path to the folder of color images."
)

parser.add_argument(
    "--depth-input",
    type=str,
    default= "./train_data/HHA/" ,
    help="Path to the folder of HHA images."
)


parser.add_argument(
    '--gt_path',
    type=str,
    default='./train_data/gt/',
    help='the scribble_foreground images path'
)

parser.add_argument(
    '--mask_path',
    type=str,
    default='./train_data/mask/',
    help='the scribble_background images path'
)

parser.add_argument(
    '--output_path',
    type=str,
    default='./train_data/superpixel/',
    help='the output path of pseudo labels'
)

def main(args: argparse.Namespace) -> None:

    if not os.path.isdir(args.input):
        color_targets = [args.input]
    else:
        color_targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        color_targets = [os.path.join(args.input, f) for f in color_targets]

    if not os.path.isdir(args.depth_input):
        depth_targets = [args.depth_input]
    else:
        depth_targets = [
            f for f in os.listdir(args.depth_input) if not os.path.isdir(os.path.join(args.depth_input, f))
        ]
        depth_targets = [os.path.join(args.depth_input, f) for f in depth_targets]

    color_targets.sort()
    depth_targets.sort()

    # 获取彩色图文件名（不含扩展名）
    color_names = [os.path.splitext(os.path.basename(f))[0] for f in color_targets]
    # 获取深度图文件名（不含扩展名）
    depth_names = [os.path.splitext(os.path.basename(f))[0] for f in depth_targets]

    # 筛选出名称相同的彩色图和深度图
    common_names = set(color_names).intersection(set(depth_names))
    new_color_targets = [f for f in color_targets if os.path.splitext(os.path.basename(f))[0] in common_names]
    new_depth_targets = [f for f in depth_targets if os.path.splitext(os.path.basename(f))[0] in common_names]

    assert len(new_color_targets) == len(new_depth_targets), "Number of color images and depth images must be the same."

    for color_t, depth_t in zip(new_color_targets, new_depth_targets):
        color_base = os.path.splitext(os.path.basename(color_t))[0]
        depth_base = os.path.splitext(os.path.basename(depth_t))[0]


        if color_base != depth_base:
            print(f"Color image '{color_t}' and depth image '{depth_t}' names do not match, skipping...")
            continue
        
        depth_t = os.path.join(os.path.dirname(depth_t), f"{color_base}{os.path.splitext(depth_t)[1]}")
        if not os.path.exists(depth_t):
            print(f"Depth image '{depth_t}' corresponding to color image '{color_t}' does not exist, skipping...")
            continue


        color_image_ori = cv2.imread(color_t)
        depth_image_ori = cv2.imread(depth_t)

        if color_image_ori is None:
            print(f"Could not load '{color_t}' as a color image, skipping...")
            continue
        if depth_image_ori is None:
            print(f"Could not load '{depth_t}' as a depth image, skipping...")
            continue

        color_image_ori = cv2.cvtColor(color_image_ori, cv2.COLOR_BGR2RGB)
        color_image = np.array(color_image_ori)
        depth_image = np.array(depth_image_ori)

        base = os.path.basename(color_t)
        base = os.path.splitext(base)[0]

        gt_path = os.path.join(args.gt_path, f"{base}.png")
        mask_path = os.path.join(args.mask_path, f"{base}.png")

        #print(f"Looking for gt image: {gt_path}")
        if not os.path.exists(gt_path):
            print(f"Gt image not found for {color_t}, skipping...")
            continue

        #print(f"Looking for mask image: {mask_path}")
        if not os.path.exists(mask_path):
            print(f"Mask image not found for {color_t}, skipping...")
            continue

        gt_ori = Image.open(gt_path)
        mask_ori = Image.open(mask_path)

        gt_ori = np.array(gt_ori)
        mask_ori = np.array(mask_ori)

        bool_mask = (gt_ori == 255) & (mask_ori == 255)
        mask_ori[bool_mask] = 0
        


        superpixel_gt_rgb, superpixel_gt_depth = prompt_extension(color_image, depth_image, gt_ori)
        superpixel_gt_rgb = (superpixel_gt_rgb[0]).numpy()
        superpixel_gt_depth = superpixel_gt_depth[0].numpy()

        superpixel_bg_rgb, superpixel_bg_depth = prompt_extension(color_image, depth_image, mask_ori)
        superpixel_bg_rgb = (superpixel_bg_rgb[0]).numpy()
        superpixel_bg_depth = superpixel_bg_depth[0].numpy()


        superpixel_gt_rgb = (superpixel_gt_rgb * 255).astype(np.uint8)
        superpixel_bg_rgb = (superpixel_bg_rgb * 255).astype(np.uint8)
        bool_mask = (superpixel_gt_rgb == 255) & (superpixel_bg_rgb == 255)
        superpixel_gt_rgb[bool_mask] = 128

        superpixel_gt_depth = (superpixel_gt_depth * 255).astype(np.uint8)
        superpixel_bg_depth = (superpixel_bg_depth * 255).astype(np.uint8)
        bool_mask = (superpixel_gt_depth == 255) & (superpixel_bg_depth == 255)
        superpixel_gt_depth[bool_mask] = 128

        bool_mask = ((superpixel_gt_rgb == 255) & (superpixel_gt_depth != 255)) | \
                    ((superpixel_gt_rgb != 255) & (superpixel_gt_depth == 255)) | \
                    ((superpixel_gt_rgb == 0) & (superpixel_gt_depth != 0)) | \
                    ((superpixel_gt_rgb != 0) & (superpixel_gt_depth == 0))
        
        superpixel_gt_rgb[bool_mask] = 128


        filename = f"{base}.png"
        cv2.imwrite(os.path.join(args.output_path, filename), superpixel_gt_rgb )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)    


    
