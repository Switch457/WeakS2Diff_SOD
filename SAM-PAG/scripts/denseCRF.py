#!/usr/bin/python

'''
denseCRF finetune
'''

import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import os
import argparse


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_path",
    type=str,
    default= "./train_data/img",
    help="Path to color images."
    )

parser.add_argument(
    "--hha_path",
    type=str,
    default= "./train_data/HHA",
    help="Path to HHA images."
    )


parser.add_argument(
    "--sal_path",
    type=str,
    default= "./train_data/output",
    help="Path to pseudo annotations pending CRF optimization."
    )

parser.add_argument(
    "--output_path",
    type=str,
    default= "./train_data/crf"
    )

def main(args: argparse.Namespace) -> None:
        
        
        files = os.listdir(args.input_path)
        files.sort()
        for file in files:
            #print(file)
            if (os.path.isfile(args.input_path+"/"+file)):

                img = cv2.imread(args.input_path+'/'+file, 1)
                hha = cv2.imread(args.hha_path + '/' + file, 1)
                if img is None or hha is None:
                    print('skip', file)
                    continue
                
                file_label = file.split('.')[0] + '_fused.png'
                annos = cv2.imread(args.sal_path+'/'+file_label, 0)
                file_output = file.split('.')[0] + '_crf.png'
                output = args.output_path+'/'+file_output

                # 将 255 映射为 1
                annos = annos // 255  # 将 255 映射为 1，0 保持不变  ###
                EPSILON = 1e-8

                M = 2  # salient or not
                tau = 1.05
                
                d = dcrf.DenseCRF2D(annos.shape[1], annos.shape[0], M)

                
                #anno_norm = annos / 255.
                anno_norm = annos.astype(np.float32) / 255.0

                n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))#-np.log(1.0 - anno_norm + EPSILON)
                p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

                U = np.zeros((M, annos.shape[0] * annos.shape[1]), dtype='float32')
                U[0, :] = n_energy.flatten()
                U[1, :] = p_energy.flatten()

                prob_map = np.stack([1.0-anno_norm, anno_norm], axis = 0)
                #energy = -np.log(prob_map + EPSILON)
                unary = unary_from_labels(annos, n_labels=2, gt_prob=0.7, zero_unsure=False)
                d.setUnaryEnergy(unary)#(U)

                d.addPairwiseGaussian(sxy=3, compat=3)
                d.addPairwiseBilateral(sxy=30, srgb=5, rgbim=img, compat=5)
                d.addPairwiseBilateral(sxy=15, srgb=5, rgbim=hha, compat=3)

                # Do the inference
                infer = np.array(d.inference(10)).astype('float32')  # number of the inferences
                res = infer[1,:]

                res = res * 255
                res = res.reshape(img.shape[:2])

                cv2.imwrite(output, res.astype('uint8'))
                print(output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)