# SPIdepth — Curriculum Learning for Monocular Depth Estimation

This repository contains my extended implementation of the **SPIdepth** network for **Monocular Depth Estimation (MDE)**, developed during my research internship.  
It is based on the original SPIdepth paper and codebase:

> **SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation (CVPR 2025)**  
> Original repository: https://github.com/Lavreniuk/SPIdepth

My work expands this project by integrating **Curriculum Learning (CL)** into both the *self-supervised* and *supervised* training pipelines, as well as adding **MidAir dataset support** through a custom PyTorch dataloader.

---

# 🚀 Project Overview

SPIdepth is a state-of-the-art network for self-supervised monocular depth estimation.  
In this repository, I introduce several major enhancements:

### ✅ **1. Curriculum Learning (CL) Training Pipeline**
I added two new training scripts at the project root:

- `train_cl.py` — Curriculum Learning training for **self-supervised MDE**  
- `trainer_cl.py` — CL-based trainer replacing the original `trainer.py`

These scripts incorporate configurable pacing strategies (e.g., *linear pacing*) and additional CL hyperparameters, including:

--a 0.8
--p 1
--pacing linear


---

### ✅ **2. MidAir Dataset Support (Custom Dataloader)**
The original SPIdepth implementation only supported **KITTI** and **Cityscapes**.  
I created a full PyTorch dataloader for the **MidAir aerial dataset**, enabling:

- loading RGB images  
- loading float16-encoded depth maps  
- dataset-specific augmentation  
- support for 384×384 inputs  
- compatibility with both standard and Curriculum Learning pipelines  

The dataset can be selected via:

--dataset midair


---

### ✅ **3. Curriculum Learning for Supervised Fine-Tuning**
Inspired by the original supervised fine-tuning script (`train_ft_SQLdepth.py`), I implemented:

- `train_ft_SQLdepth_midair.py` — standard supervised fine-tuning on MidAir  
- `train_cl_sl_ft_midair.py` — **supervised Curriculum Learning** training on MidAir  

These scripts allow CL strategies to be applied not only to self-supervised training but also to fine-tuning with ground-truth depth maps.

---

# 🧪 Training

## ▶️ **Self-Supervised Training (Original SPIdepth)**

Example (KITTI):
```bash
python train.py ./args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```

## ▶️ Self-Supervised Curriculum Learning Training (New)

Example (KITTI):
```bash
python train_cl.py ./args_files/hisfog/kitti/resnet_192x640_lite_cl.txt
```

Example (MidAir):
```bash
python train_cl.py ./args_files/midair/resnet_384x384_midair_cl.txt
```

Your .txt config defines:

--dataset midair
--a 0.8
--p 1
--pacing linear

# 🧭 Fine-Tuning (Supervised)
## ▶️ Standard Supervised Fine-Tuning (MidAir)
```bash
python finetune/train_ft_SQLdepth_midair.py ./conf/midair/resnet_384.txt
```

## ▶️ Supervised Curriculum Learning (MidAir)
```bash
python finetune/train_cl_sl_ft_midair.py ./conf/midair/resnet_384_cl.txt
```

# 📊 Evaluation

Evaluation follows the original SPIdepth pipeline.

KITTI:
```bash
python evaluate_depth_config.py args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```

Cityscapes:
```bash
python tools/evaluate_depth_cityscapes_config.py args_files/args_cvnXt_H_cityscapes_finetune_eval.txt
```

MidAir:

If using MidAir, ensure your dataloader is configured in the evaluation script.

# 🖼️ Inference on Custom Images
```bash
python test_simple_SQL_config.py ./conf/cvnXt.txt
```

You can set:

--image_path <path_to_image_or_folder>

# 📚 Citation (Original SPIdepth Work)

If you use SPIdepth, please cite the original authors:

```bash
@InProceedings{Lavreniuk_2025_CVPR,
    author    = {Lavreniuk, Mykola and Lavreniuk, Alla},
    title     = {SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {874-884}
}
```

# 🙏 Acknowledgments

This work builds upon:

SPIdepth (Lavreniuk et al., CVPR 2025)

SQLdepth, the foundation of their implementation

MidAir dataset for aerial depth estimation research

My contribution focuses on expanding the original SPIdepth codebase with Curriculum Learning training strategies, MidAir dataset support, and supervised CL fine-tuning extensions.

# ✨ Author

Jorge Victor Turriate Llallire
Master in Mechatronics, Machine Vision & AI
Université Paris-Saclay
