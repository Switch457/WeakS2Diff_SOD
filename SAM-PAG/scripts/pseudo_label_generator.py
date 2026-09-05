import cv2  # type: ignore
from PIL import Image
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
import os
from pseudo_label_transform import label_from_transform
from pseudo_label_superpixel import label_from_superpixel

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

def fuse_mask(masks):
    best_mask_index =  np.argsort([(m["IoU"]) for m in masks])[-4:]
    scores = sum(masks[i]["IoU"] for i in best_mask_index)

    selected_masks = []
    weighted_prob = 0
    for i in best_mask_index:
        weighted_prob += masks[i]["IoU"] / scores * masks[i]["probabilities"]
        #combined_segmentation = (combined_prob > 0.5).astype(np.uint8)    
        selected_masks.append({
            "segmentation": masks[i]["segmentation"],
            "probabilities": weighted_prob,  # 保存融合后的概率图
        })

    
    ious = []
    for i in range(len(selected_masks)):
        for j in range(i+1, len(selected_masks)):
            iou = calculate_iou(selected_masks[i]["segmentation"], selected_masks[j]["segmentation"])
            ious.append(iou)
    average_iou = np.mean(ious)
    combined_mask = {"probabilities" : weighted_prob,
                     "iou" : average_iou
                    }
    return combined_mask

def calculate_iou(mask1, mask2):
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8), (mask1.shape[1], mask1.shape[0]))
        mask2 = mask2.astype(np.bool_)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

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
    "--output",
    type=str,
    default= "./output",
    help=(
        "Path to the directory where masks will be output. "
    )
)


parser.add_argument(
    "--model-type",
    type=str,
    default= 'vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default= './segment-anything-main/sam_vit_h_4b8939.pth',
    help="The path to the SAM checkpoint to use for mask generation."
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

      
parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    )
)

parser.add_argument(
    '--gt_path',
    type=str,
    default='./train_data/gt/',
    help='Path to the folder of foreground scribbles.'
)

parser.add_argument(
    '--mask_path',
    type=str,
    default='./train_data/mask/',
    help='Path to the folder of background scribbles.'
)

parser.add_argument(
    '--superpixel_path',
    type=str,
    default='./train_data/superpixel/',
    help='Path to the folder of extended prompts.'
)

def main(args: argparse.Namespace) -> None:

    np.random.seed(3) 
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)


    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

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

    os.makedirs(args.output, exist_ok=True)
    

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

        print(f"Processing color image '{color_t}' and depth image '{depth_t}'...")
        color_image_ori_ = cv2.imread(color_t)

        depth_image_ori = cv2.imread(depth_t)

        if color_image_ori_ is None:
            print(f"Could not load '{color_t}' as a color image, skipping...")
            continue
        if depth_image_ori is None:
            print(f"Could not load '{depth_t}' as a depth image, skipping...")
            continue

        color_image_ori = cv2.cvtColor(color_image_ori_, cv2.COLOR_BGR2RGB)

        base = os.path.basename(color_t)
        base = os.path.splitext(base)[0]

        gt_path = os.path.join(args.gt_path, f"{base}.png")
        mask_path = os.path.join(args.mask_path, f"{base}.png")
        superpixel_path = os.path.join(args.superpixel_path, f"{base}.png")

        print(f"Looking for gt image: {gt_path}")
        if not os.path.exists(gt_path):
            print(f"Gt image not found for {color_t}, skipping...")
            continue

        print(f"Looking for mask image: {mask_path}")
        if not os.path.exists(mask_path):
            print(f"Mask image not found for {color_t}, skipping...")
            continue

        gt_ori = Image.open(gt_path)
        mask_ori = Image.open(mask_path)
        width, height = gt_ori.size

        gt_ori = np.array(gt_ori)
        mask_ori = np.array(mask_ori)
        superpixel_ori = Image.open(superpixel_path)
        superpixel_ori = superpixel_ori.convert('L')
        #superpixel_ori = np.array(superpixel_ori)
       

        masks_transform = label_from_transform(color_image_ori, depth_image_ori, gt_ori, mask_ori, superpixel_ori, sam, base)
        masks_superpixel = label_from_superpixel(color_image_ori, depth_image_ori, gt_ori, mask_ori, superpixel_ori, sam, base)
        masks_all = masks_transform + masks_superpixel
        
        
        #IoU 一致性排序融合
        combined_mask = fuse_mask(masks_all)


        binary = ((combined_mask["probabilities"] > 0.5)*255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f"{base}_fused.png"), binary)

       

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    
