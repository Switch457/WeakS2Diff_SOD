# Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion

We employ the **Segment Anything Model (SAM)** to generate high-quality pseudo annotations from weak scribble supervision. The implementation is based on the official [Segment Anything](https://github.com/facebookresearch/segment-anything) repository.

### SAM-PAG Architecture

<div align="center">
  <img src="imgs\sam_pag.png" width="80%">
</div>


### Requirements

- opencv-python == 4.11.0.86
- Pillow == 11.0.0
- numpy == 2.2.4
- fast-slic == 0.4.0
- pydensecrf == 1.0
- scikit-image == 0.25.2
- segment-anything == 1.0

### Preparation

Please download the pretrained SAM checkpoint. We use **SAM ViT-H** by default:
- [sam_vit_h](https://pan.baidu.com/s/1PqR1NYCQpUGOk7TxANZdSA?pwd=kbnf)

Download and prepare the training data as follows:

```text
train_data/
├── img/          # RGB images
├── HHA/          # HHA depth images
├── gt/           # foreground scribbles
├── mask/         # background scribbles
└── superpixel/       # extended superpixel prompts
```

**Datasets**: [Download Link](https://pan.baidu.com/s/1okA-U8ceJDQIW44ai4cuyA?pwd=wuch)

### 1. Convert depth map to HHA

Download depth2HHA.zip and unzip it, run depth2HHA.m to convert depth map to HHA.
You can also directly use the generated HHA map in HHA folder.

### 2. Prompt Extension

Generate the superpixel-based extended prompts using:

```bash
python prompt_extension.py
```

You can also directly use the generated extended prompts in superpixel folder.

### 3. Generate Pseudo Annotations

Run:

```bash
python pseudo_label_generator.py \
    --input /path/to/img \
    --depth-input /path/to/HHA \
    --gt_path /path/to/gt \
    --mask_path /path/to/mask \
    --superpixel_path /path/to/superpixel \
    --checkpoint /path/to/sam_vit_h_4b8939.pth \
    --output /path/to/output
```

### 4. DenseCRF Refinement

The generated pseudo annotations can be further refined using DenseCRF:

```bash
python denseCRF.py \
    --input_path /path/to/img \
    --hha_path /path/to/HHA \
    --sal_path /path/to/output \
    --output_path /path/to/crf_output
```

We also provide our refined pseudo annotations in pseudo_gt folder.

## Acknowledgement
   
This work is built upon [segment anything](https://github.com/facebookresearch/segment-anything).
We sincerely thank the authors for their great contributions.
