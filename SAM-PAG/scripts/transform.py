import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import random
import torch


def cv_flip(image): #水平翻转
    flipped_horizontal = cv2.flip(image, 1)
    return flipped_horizontal


def flip_points(input_point, width):

    flipped = input_point.clone()

    flipped[:, 0] = width - 1 - input_point[:, 0]
    return flipped


def cv_resize(image, width, height): #缩放
    resized_image = cv2.resize(image,(width, height)) 
    return resized_image

def resize_points(input_point, scale=2):

    # 先转为 float 做除法，再取整，最后转回原 dtype
    return (input_point.float() / scale).floor().to(input_point.dtype)

def cv_rotation(image, i): #旋转90
    rotated_90 = np.rot90(image, i) # -1顺时针 1逆时针
    return rotated_90

def rotation_points(input_point, width):
    rotated = torch.empty_like(input_point)
    rotated[:, 0] = input_point[:, 1]          # x' = y
    rotated[:, 1] = width - 1 - input_point[:, 0]  # y' = width - 1 - x
    return rotated



def randomPeper(img):
    img = np.array(img, dtype=np.uint8)  # 将输入图像转换为NumPy数组
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])  # 计算要添加的噪声点数量
    # 随机生成噪声点的坐标
    randX = np.random.randint(0, img.shape[0], size=noiseNum)
    randY = np.random.randint(0, img.shape[1], size=noiseNum)
    # 随机生成噪声点的颜色，0表示黑色，255表示白色
    noiseColor = np.random.choice([0, 255], size=noiseNum)
    # 将噪声添加到图像上
    img[randX, randY] = noiseColor
    return Image.fromarray(img)

def colorEnhance(image):
    # 随机生成亮度调整强度，范围在0.8到1.2之间
    bright_intensity = random.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)

    # 随机生成对比度调整强度，范围在0.8到1.2之间
    contrast_intensity = random.uniform(0.8, 1.2)
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)

    # 随机生成色彩饱和度调整强度，范围在0.8到1.2之间
    color_intensity = random.uniform(0.8, 1.2)
    image = ImageEnhance.Color(image).enhance(color_intensity)

    # 随机生成锐度调整强度，范围在0.8到1.2之间
    sharp_intensity = random.uniform(0.8, 1.2)
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)

    return image

def color_enhance_np(img_rgb,  # uint8 ndarray, RGB
                     bright_range=(0.8, 1.2),
                     contrast_range=(0.8, 1.2),
                     sat_range=(0.8, 1.2),
                     sharp_range=(0.8, 1.2)):
    """
    对 uint8 RGB ndarray 做亮度、对比度、饱和度、锐度的随机增强
    返回 uint8 ndarray
    """
    img = img_rgb.astype(np.float32) / 255.0          # 转到 0-1 float32

    # 1. 亮度
    bright = np.random.uniform(*bright_range)
    img = np.clip(img * bright, 0, 1)

    # 2. 对比度 (以灰度均值为中心)
    contrast = np.random.uniform(*contrast_range)
    mean = np.mean(img)
    img = np.clip((img - mean) * contrast + mean, 0, 1)

    # 3. 饱和度 (HSV 空间)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat_scale = np.random.uniform(*sat_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 4. 锐度 (USM 简单版)
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    sharp_scale = np.random.uniform(*sharp_range)
    img = np.clip(img + (img - blur) * sharp_scale, 0, 1)

    return (img * 255).astype(np.uint8)

def randomGaussian(image, mean=0.1, sigma=0.35):
    # 将输入图像转换为NumPy数组
    img = np.asarray(image, dtype=np.float32)
    # 生成高斯噪声并添加到图像上
    noise = np.random.normal(mean, sigma, img.shape)
    img += noise
    # 确保图像数据在有效范围内
    img = np.clip(img, 0, 255)
    # 将NumPy数组转换回图像，并返回
    return Image.fromarray(np.uint8(img))

def drawing(bin_img,
            text=None,
            top_margin=100,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=3.3,
            font_color=(0, 0, 0),   # 黑色字
            font_thickness=7):
    """
    给二值图加白色底板并在上方留空白写文字
    :param bin_img:     单通道二值图，dtype=uint8
    :param text:        要写的字符串
    :param top_margin:  上方预留空白高度（像素）
    :param font:        OpenCV 字体
    :param font_scale:  字体大小
    :param font_color:  字体颜色，默认黑色
    :param font_thickness: 笔画粗细
    :return:            处理后的 BGR 图（可直接 imwrite）
    """
    if len(bin_img.shape) == 3:
        raise ValueError('请输入单通道二值图')

    h, w = bin_img.shape

    # 1. 创建白色底板
    canvas = np.ones((h + top_margin, w, 3), dtype=np.uint8) * 255

    # 2. 把二值图贴到底板下方（单通道→三通道）
    canvas[top_margin:, :] = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 3. 计算文字位置（水平居中）
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    org = ((w - text_w) // 2, (top_margin + text_h) // 2)  # 垂直居中微调
    

    # 4. 写字
    cv2.putText(canvas, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return canvas


"""
gt_path = '/media/swq/8ea992ba-afb3-44ce-8a80-69ff10668360/swq/WS-SAM-main/dataset/TrainDataset/train_data/gt/1_02-02-40.png'
gt = Image.open(gt_path)
print(gt.size)
width, height = gt.size
plt.imshow(gt)
plt.axis('off')  # 关闭坐标轴
plt.show()
gt = np.array(gt)
gt = cv_resize(gt,width,height)
print(gt.shape)
gt = Image.fromarray(gt)
print(gt.size)
"""
