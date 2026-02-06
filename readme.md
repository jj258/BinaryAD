<h2 align="center">BinaryAD: Efficient Image Anomaly Detection via Binarized Representations</h2>

![Pipeline](pipeline.png)

## Abstract

The steadily increasing complexity of Transformer-based image anomaly detection models poses significant challenges for deployment on resource-constrained edge devices. Although pruning, knowledge distillation, and low-bit quantization have achieved notable success in classification networks, directly applying full-model binarization or uniform low-bit quantization to Transformer-based anomaly detectors often leads to severe degradation in pixel-level anomaly localization performance. A key observation motivating this work is that the prototype extraction and reconstruction objectives commonly adopted in Transformer-based anomaly detection models are highly sensitive to quantization noise in the attention and MLP modules.
In this paper, we propose BinaryAD, a module-selective binarization framework that learns compact binarized representations in the Transformer decoder to enable lightweight yet accurate anomaly detection models. Specifically, BinaryAD selectively binarizes only the most computationally intensive components, namely the self-attention and MLP layers in the Transformer decoder, while preserving full-precision feature extractors to maintain sufficient representational capacity.
We instantiate BinaryAD on the representative INP and Dinomaly models and conduct extensive experiments on standard anomaly detection benchmarks. Experimental results demonstrate that, across datasets, BinaryAD reduces model size by up to 6.53$\times$ and FLOPs by up to 7.99$\times$. More importantly, under optimal configurations, BinaryAD incurs only a minimal performance gap compared to full-precision baselines, with I-AUROC degradation of at most 1.1\%, while still retaining competitive pixel-level anomaly localization accuracy.
These results indicate that, when properly designed, module-selective binarization offers a practical and effective pathway for deploying advanced Transformer-based anomaly detection architectures on low-power, real-time industrial edge hardware.

## BinaryAD_INP 

---

### Installation

Create an environment and install dependencies for **BinaryAD_INP**:

```bash
conda create -n binaryad_inp python=3.10 -y
conda activate binaryad_inp
pip install -r requirements.txt
```

### Datasets

MVTec-AD 

```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```

VisA

```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```

Real-IAD

```
|-- Real-IAD
    |-- realiad_1024
        |-- audiokack
        |-- bottle_cap
        |-- ....
    |-- realiad_jsons
        |-- realiad_jsons
        |-- realiad_jsons_sv
        |-- realiad_jsons_fuiad_0.0
        |-- ....
```

------

### Run

1) Single-Class 

```
python Binary_INP_Single_Class.py \
  --dataset MVTec-AD \
  --data_path ./datasets/mvtec \
  --phase train \
  --clb_stage 1
```

------

2) Few-Shot 

```
python Binary_INP_Few_Shot.py \
  --dataset MVTec-AD \
  --data_path ./datasets/mvtec \
  --phase train \
  --shot 4 \
  --clb_stage 1
```

------

3) Multi-Class

```
python Binary_INP_Multi_Class.py \
  --dataset MVTec-AD \
  --data_path ./datasets/mvtec \
  --phase train \
  --clb_stage 1
```

------

### CLB Training

#### Stage 1 (Attention binarization)

Run any script with:

```
--clb_stage 1
```

#### Stage 2 (Attention + MLP binarization)

Multi-class / Few-shot

```
python Binary_INP_Multi_Class.py \
  --dataset MVTec-AD \
  --data_path ./datasets/mvtec \
  --phase train \
  --clb_stage 2 \
  --load_checkpoint /path/to/stage1_model.pth
```

Single-class

Single-class stage-2 supports **auto-resolving** stage-1 checkpoints for each category, based on:

- `--save_dir`
- `--load_checkpoint` (used as a prefix name)
- `dataset/encoder/resize/crop/INP_num`
- `args.item` and `model.pth`

Example:

```
python Binary_INP_Single_Class.py \
  --dataset MVTec-AD \
  --data_path ./datasets/mvtec \
  --phase train \
  --clb_stage 2 \
  --load_checkpoint INP-Former-Single-Class
```

---

## BinaryAD_Dinomaly 

### Installation

Create an environment and install dependencies for **BinaryAD-Dinomaly**:

```bash
conda create -n binaryad_dino python=3.10 -y
conda activate binaryad_dino
pip install -r requirements.txt
```

### Datasets

```
Refer to the datasets of binaryAD_INP
```

### Run

```
Refer to the run of binaryAD_INP
```








