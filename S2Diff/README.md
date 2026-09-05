# Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion

This repository contains the **model implementation** for the paper:

> **Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion**

[Paper Link](https://arxiv.org/pdf/2607.15041)

## Network Architecture

<div align="center">
  <img src="imgs\network.png" width="80%">
</div>



## Requirements

- Python == 3.10
- torch == 2.5.1
- torchvision == 0.20.1
- numpy == 2.0.2
- accelerate == 1.2.1
- einops == 0.8.0
- opencv-python == 4.10.0.84
- tqdm == 4.67.1
- timm == 1.0.12
- mamba-ssm == 2.2.3.post2
- causal-conv1d == 1.5.0.post7
- mmcv == 2.0.0rc3
- wandb == 0.19.1
- matplotlib == 3.10.0
- huggingface-hub == 0.31.1

## Dataset

We provide the RGB-D salient object detection datasets, please download them and modify the corresponding dataset paths in the configuration files.

**Datasets**: [Download Link](https://pan.baidu.com/s/1okA-U8ceJDQIW44ai4cuyA?pwd=wuch)

The current evaluation supports:

- DUT-RGBD
- LFSD
- NJU2K
- NLPR
- SIP
- SSD
- STERE

## Training

1. Download the pretrained backbone in pretrained_weights:

   - [pvt_v2_b4_m](https://pan.baidu.com/s/1Sl1SHp4QcRIhBjLttwqaPQ?pwd=hxc3)

2. Set your dataset paths in the configuration file: 
    - config\dataset_352x352.yaml.

3. Run:

```bash
accelerate launch train.py --config config/camoDiffusion_352x352.yaml --num_epoch=150 --batch_size=8 --gradient_accumulate_every=1
```

## Testing

Run:

```bash
accelerate launch sample.py --config config/camoDiffusion_352x352.yaml --results_folder ${RESULT_SAVE_PATH} --checkpoint ${CHECKPOINT_PATH} --num_sample_steps 10
```

## Saliency Maps

We provide the predicted saliency maps of our method and other weakly-supervised methods.

- **S^2Diff (ours)**: [Download Link](https://pan.baidu.com/s/16k37ZpYFP1s5xD9XMlPXPg?pwd=v8mj)
- **Other weakly-supervised methods**: [Download Link](https://pan.baidu.com/s/1DJrT1qqnbouljqS8HX7-Cw?pwd=8dt4)

<div align="center">
  <img src="imgs\comparisons_1.png" width="80%">
</div>

<div align="center">
  <img src="imgs\comparisons_2.png" width="80%">
</div>

## Trained Model
Download our trained model from [checkpoint](https://pan.baidu.com/s/16k37ZpYFP1s5xD9XMlPXPg?pwd=v8mj).

## Evaluation
 We recommend using the following toolkit for evaluation: [Evaluation Tool](https://github.com/lartpang/PySODEvalToolkit).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{si2026weakly,
  title={Weakly-Supervised {RGB-D} Salient Object Detection via {SAM}-driven Pseudo Annotation and State Space Interaction-based Diffusion},
  author={Si, Wenqi and Li, Gongyang and Shi, Shixiang and Lin, Weisi},
  journal={IEEE Transactions on Multimedia},
  year={2026},
}
```

## Acknowledgement
   
This work is built upon [Camodiffusion](https://github.com/Rapisurazurite/CamoDiffusion) and [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet).
We sincerely thank the authors for their great contributions.

