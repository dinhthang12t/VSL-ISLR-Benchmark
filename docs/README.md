# Benchmark Suite: Temporal Modeling for Large-Scale Vietnamese Isolated Sign Language Recognition

Official reproduction repository for the benchmark paper:
**"A Comparative Study of Temporal Modeling for Large-Scale Vietnamese Isolated Sign Language Recognition"**

---

## 1. Overview & Benchmark Scope

Isolated Sign Language Recognition (ISLR) in Vietnamese presents distinctive challenges including fine-grained hand gestures, rapid non-manual motion, and severe performance degradation under unseen signers. This benchmark provides the first systematic, controlled empirical evaluation of **18 model architectures across 4 distinct paradigms**:

1. **2D CNN + Temporal Aggregation**: Frame feature extraction paired with temporal Mean Pooling, GRU, or LSTM recurrent aggregation heads (ResNet-18, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny).
2. **3D CNN Spatiotemporal Models**: End-to-end spatiotemporal convolutions (R3D-18, Inception-I3D, X3D-M, C3D, R(2+1)D).
3. **Video Transformers**: Spatiotemporal attention and factorized video transformers (Swin3D-T, TimeSformer, ViViT).
4. **Pose-based Spatiotemporal Graphs**: Temporal Graph Convolutional Networks (TGCN) operating on 55 skeletal keypoints extracted via MediaPipe, MMPose, and ViTPose.

---

## 2. Evaluation Protocols & Dataset Splits

The benchmark evaluates all paradigms under two standardized evaluation protocols:

```
data/
├── splits_original_1182/        # Protocol 1: Class-stratified with signer overlap
│   ├── ISLR_promax.csv          # 7,664 videos (Train: 4,154, Val: 1,755, Test: 1,755)
│   ├── 100collins.csv           # Frequent vocabulary subset (100 classes)
│   └── 300collins.csv           # Frequent vocabulary subset (300 classes)
│
└── splits_signer_indep_447/     # Protocol 2: Signer-independent (Table 11)
    ├── ISLR_447_benchmark.csv   # 4,252 videos (Train: 2,344, Val: 1,021, Test: 887)
    ├── class_split_audit.csv    # Per-class retention and exclusion audit
    └── signer_assignment.csv    # Disjoint partition of 50 signers (25 / 13 / 12)
```

### Protocol 1: Class-Stratified Split (1,182 Classes)
- **Vocabulary**: 1,182 distinct Vietnamese sign classes.
- **Dataset Size**: 7,664 videos from 57 signers (Train: 4,154, Validation: 1,755, Test: 1,755).
- **Properties**: Signers overlap across splits. Evaluates class discriminative power under known signer styles.

### Protocol 2: Signer-Independent Split (447 Classes)
- **Vocabulary**: 447 classes selected via combinatorial local search to ensure complete class representation in every split.
- **Dataset Size**: 4,252 videos from 50 disjoint signers.
- **Split Distribution**:
  - **Train**: 2,344 videos from 25 disjoint signers.
  - **Validation**: 1,021 videos from 13 disjoint signers.
  - **Test**: 887 videos from 12 disjoint signers.
- **Properties**: **Zero signer overlap** ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$). Evaluates true out-of-domain generalization to unseen individuals.

---

## 3. Repository Directory Structure

```
VSL-ISLR-Benchmark/
├── README.md                      # General landing page
├── docs/
│   └── README.md                  # This detailed reproduction and benchmark manual
├── requirements.txt               # PIP package dependencies
├── environment.yml                # Conda environment configuration
│
├── data/                          # Benchmark split manifests (CSV metadata only)
│   ├── splits_original_1182/
│   └── splits_signer_indep_447/
│
├── src/
│   ├── models/                    # Model definitions for all 18 architectures
│   │   ├── cnn2d_temporal.py      # ResNet/EfficientNet/ConvNeXt + Mean/GRU/LSTM
│   │   ├── cnn3d.py               # R3D-18, I3D, X3D-M, C3D, R(2+1)D
│   │   ├── transformers.py        # Swin3D-T, TimeSformer, ViViT
│   │   └── tgcn.py                # Pose-based Spatiotemporal GCN (3 Residual Blocks)
│   │
│   ├── datasets/                  # PyTorch Dataset loaders
│   │   ├── feature_dataset.py     # 2D CNN pre-extracted frame features loader
│   │   ├── video_dataset.py       # Raw RGB video loader with short-edge resize & crop
│   │   ├── pose_dataset.py        # 55-keypoint skeletal sequence loader
│   │   └── transforms.py          # Kinetics/ImageNet normalizations, spatial croppings
│   │
│   └── evaluation/                # Reviewer verification and audit tools
│       ├── metrics.py             # Top-k, Macro-F1, Head/Medium/Tail accuracy
│       ├── reproduce_tables.py    # Per-model evaluation & JSON prediction generator
│       └── verify_all_models.py   # Automated batch auditor for Table 4 and Table 11
│
├── scripts/                       # Training and evaluation runner scripts
└── results/                       # Verified sample-level predictions (JSON reports)
```

---

## 4. Environment Setup

### Using Conda (Recommended)

```bash
# 1. Create environment
conda env create -f environment.yml

# 2. Activate environment
conda activate vsl-benchmark

# 3. Verify PyTorch and CUDA availability
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

### Using Pip

```bash
pip install -r requirements.txt
```

---

## 5. Published Benchmark Results

### Table 4: Original Class-Stratified Benchmark (1,182 Classes, Signer Overlap)

| Category | Model Architecture | Top-1 (%) | Top-5 (%) | Top-10 (%) | Audit Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2D CNN + Mean** | ResNet-18 + Mean | 45.24 | 52.82 | 55.90 | **EXACT MATCH** |
| | ResNet-50 + Mean | **46.89** | **54.64** | 57.32 | **EXACT MATCH** |
| | EfficientNet-B0 + Mean | 46.32 | 54.36 | **57.61** | **EXACT MATCH** |
| | ConvNeXt-Tiny + Mean | 45.87 | 54.19 | 57.09 | **EXACT MATCH** |
| **2D + Temporal** | ResNet-18 + GRU | 36.87 | 49.63 | 54.99 | **EXACT MATCH** |
| | ResNet-50 + GRU | 37.66 | 50.60 | 55.21 | **EXACT MATCH** |
| | EfficientNet-B0 + GRU | **39.03** | **51.40** | **55.44** | **EXACT MATCH** |
| | ConvNeXt-Tiny + GRU | 38.63 | 49.91 | 54.87 | **EXACT MATCH** |
| *(Table 8)* | ResNet-50 + LSTM | 38.63 | 48.89 | 53.56 | **EXACT MATCH** |
| **3D CNN** | R3D-18 | 48.09 | 64.16 | 68.32 | **MATCH** |
| | I3D | 50.26 | 63.65 | 68.38 | **EXACT MATCH** |
| | X3D-M | **54.76** | **68.03** | **71.91** | **MATCH** |
| | C3D | 47.86 | 60.57 | 64.56 | **EXACT MATCH** |
| **Transformer** | Swin3D-T | **60.51** | **71.40** | **75.04** | **EXACT MATCH** |
| **Pose-based** | MediaPipe + TGCN | 45.76 | 60.62 | 65.69 | **EXACT MATCH** |
| | MMPose + TGCN | 46.64 | 60.94 | 66.49 | **EXACT MATCH** |
| | ViTPose + TGCN | **49.06** | **62.75** | **66.77** | **EXACT MATCH** |

---

### Table 11: Signer-Independent Benchmark (447 Classes, Unseen Signers)

| Paradigm | Model Architecture | Top-1 | Top-5 | Top-10 | Macro-F1 | Head Top-1 | Med Top-1 | Tail Top-1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D CNN** | ResNet-18 + Mean | 1.58% | 6.09% | 11.39% | 0.90% | 2.58% | 0.75% | 0.52% |
| | ResNet-50 + Mean | 2.25% | 7.67% | 12.18% | 0.98% | 3.52% | 1.87% | 0.00% |
| | EfficientNet-B0 + Mean | 1.35% | 5.98% | 10.37% | 1.12% | 1.64% | 1.12% | 1.04% |
| | ConvNeXt-Tiny + Mean | 1.13% | 6.09% | 9.70% | 0.58% | 1.64% | 1.12% | 0.00% |
| **2D + GRU** | ResNet-18 + GRU | 0.68% | 3.61% | 6.43% | 0.30% | 1.41% | 0.00% | 0.00% |
| | ResNet-50 + GRU | 1.01% | 4.51% | 6.65% | 0.58% | 1.41% | 1.12% | 0.00% |
| | EfficientNet-B0 + GRU | 1.24% | 3.38% | 6.88% | 0.58% | 2.11% | 0.75% | 0.00% |
| | ConvNeXt-Tiny + GRU | 1.01% | 3.95% | 6.76% | 0.38% | 2.11% | 0.00% | 0.00% |
| **3D CNN** | R3D-18 | 14.21% | 29.76% | 38.78% | 8.68% | 23.00% | 7.46% | 4.15% |
| | I3D | 17.81% | 37.54% | 46.11% | 11.33% | 30.05% | 8.58% | 3.63% |
| | X3D-M | 18.04% | 29.65% | 33.71% | 7.83% | 31.22% | 10.07% | 0.00% |
| | R(2+1)D | 13.19% | 19.17% | 22.44% | 5.24% | 26.06% | 1.49% | 1.04% |
| **Transformer**| Swin3D-T | **31.00%**| **55.24%**| **66.40%**| **21.28%**| **44.13%**| **25.37%**| **9.84%**|
| | TimeSformer | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| | ViViT | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Pose-based** | MediaPipe + TGCN | 14.43% | 29.76% | 38.56% | 9.50% | 20.19% | 12.31% | 4.66% |
| | MMPose + TGCN | 13.42% | 30.10% | 41.26% | 8.22% | 19.72% | 10.07% | 4.15% |
| | ViTPose + TGCN | 15.45% | 31.91% | 41.04% | 9.89% | 21.36% | 14.55% | 3.63% |

> **Note**: Head, medium, and tail classes are defined based on frequency in the training split: $\ge 6$, $3–5$, and $1–2$ videos/class, corresponding to 182, 135, and 130 classes (containing 426, 268, and 193 test clips), respectively. Groupwise Top-1 accuracy is evaluated over test samples of the corresponding ground-truth subset when the model classifies across all 447 classes. Macro-F1 is averaged across all 447 classes.

---

## 6. How to Reproduce and Verify (Reviewer Guide)

### Option A: Fast Verification Suite (All Models)

Reviewers can verify the mathematical reproducibility of all models across Table 4 and Table 11 with a single command:

```bash
# Verify Table 4 (1182 classes, 17 models)
python src/evaluation/verify_all_models.py --table 4

# Verify Table 11 (447 classes, 18 models)
python src/evaluation/verify_all_models.py --table 11

# Verify both protocols and print summary tables
python src/evaluation/verify_all_models.py --table all
```

This generates detailed sample-level predictions and verification audits in `results/`.

---

### Option B: Evaluating Individual Models via Live Inference

Reviewers can run live forward inference on the test set using any checkpoint:

#### 1. 2D CNN + Temporal Models (Table 4 or Table 11)
```bash
# Protocol 1 (1182 classes): ResNet-50 + Mean
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model resnet50_mean \
    --checkpoint checkpoints/resnet50_mean.pth \
    --data_dir /path/to/features/rgb_resnet50_64_iter \
    --output_json results/table4_resnet50_mean.json

# Protocol 1: EfficientNet-B0 + GRU
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model efficientnet_b0_gru \
    --checkpoint checkpoints/efficientnet_b0_gru.pth \
    --data_dir /path/to/features/rgb_efficientnet_b0_64_iter \
    --output_json results/table4_efficientnet_b0_gru.json
```

#### 2. 3D CNN Models (Table 4 or Table 11)
```bash
# Protocol 1: Inception-I3D
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model i3d \
    --checkpoint checkpoints/i3d_best.pth \
    --data_dir /path/to/video_raw \
    --output_json results/table4_i3d.json

# Protocol 1: X3D-M
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model x3d_m \
    --checkpoint checkpoints/x3d_best.pth \
    --data_dir /path/to/video_raw \
    --output_json results/table4_x3d.json
```

#### 3. Transformer Models (Table 4 or Table 11)
```bash
# Protocol 1: Swin3D-T
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model swin3d_t \
    --checkpoint checkpoints/swin3t_best.pth \
    --data_dir /path/to/video_raw \
    --output_json results/table4_swin3d_t.json
```

#### 4. Pose-based TGCN Models (Table 4 or Table 11)
```bash
# Protocol 1: ViTPose + TGCN
python src/evaluation/reproduce_tables.py \
    --table 4 \
    --model vitpose_tgcn \
    --checkpoint checkpoints/vitpose_tgcn.pth \
    --data_dir /path/to/pose_features/vitpose \
    --output_json results/table4_vitpose_tgcn.json
```

---

### Option C: Evaluating from Prediction Files (`.npz`, `.csv`, `.txt`)

To reproduce numbers instantly without re-running full GPU inference:

```bash
python src/evaluation/reproduce_tables.py \
    --table 11 \
    --model swin3d_t \
    --from_predictions path/to/test_predictions.npz \
    --output_json results/table11_swin3d_t.json
```

---

## 7. JSON Output Format

When `--output_json` is specified, the script saves a structured JSON audit report:

```json
{
  "model": "swin3d_t",
  "table": 11,
  "split": "signer_independent_447",
  "num_classes": 447,
  "num_test_samples": 887,
  "metrics": {
    "top1": 31.00,
    "top5": 55.24,
    "top10": 66.40,
    "macro_f1": 21.28,
    "head_top1": 44.13,
    "medium_top1": 25.37,
    "tail_top1": 9.84
  },
  "published_reference": {
    "top1": 31.00,
    "top5": 55.24,
    "top10": 66.40,
    "macro_f1": 21.28,
    "head_top1": 44.13,
    "medium_top1": 25.37,
    "tail_top1": 9.84
  },
  "status": "VERIFIED_EXACT_MATCH",
  "predictions": [
    {
      "video_id": "test_video_001.mp4",
      "y_true": 142,
      "y_pred_top1": 142,
      "correct": true
    }
  ]
}
```
