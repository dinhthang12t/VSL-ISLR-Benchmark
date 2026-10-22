# VSL-ISLR Benchmark

Resources for **A Comparative Study of Temporal Modeling for Large-Scale Vietnamese Isolated Sign Language Recognition**.

**Repository:** [dinhthang12t/VSL-ISLR-Benchmark](https://github.com/dinhthang12t/VSL-ISLR-Benchmark).

Implementation files are maintained under `src/` in this repository. Data-access links and experiment resources are documented below as they become available.

> Before publication, complete the fields marked `[TO COMPLETE]` and ensure that the resources described below are actually included or linked.

## Overview

This project compares temporal modeling approaches for Vietnamese isolated sign language recognition (ISLR), including:

- 2D CNN features with mean pooling or GRU aggregation.
- 3D CNN-based spatiotemporal models.
- Video transformers.
- Pose-based temporal graph models.

The repository uses a simple structure. Implementation files are organized under `src/`; installation, configuration and experiment instructions are documented in this README.

```text
VSL-ISLR-Benchmark/
├── README.md
├── DATA_LICENSE.md
└── src/
```

Code license: `[TO COMPLETE: selected code license and its location]`.

## Evaluation protocols

### Original class-stratified protocol

The original experiments use class-stratified splits with signer overlap. These results compare models under that protocol and do not establish generalization to unseen signers.

The current corrected metadata contains **7,664 videos across 1,182 classes and 57 signer IDs**. The earlier manuscript reports 7,665 videos; identify the exact manifest version used for the original experiments and document the duplicate-ID correction. Do not present the original results as having been recomputed unless they have been rerun.

### Signer-independent protocol

The additional benchmark contains **4,252 videos across 447 classes and 50 signers**. The split package is dated **9 October 2026**. Use this 447-class partition consistently; do not mix its manifests with the alternative 572-class partition.

| Split | Videos | Signers | Classes | Video share |
|---|---:|---:|---:|---:|
| Training | 2,344 | 25 | 447 | 55.13% |
| Validation | 1,021 | 13 | 447 | 24.01% |
| Test | 887 | 12 | 447 | 20.86% |
| Total | 4,252 | 50 | 447 | 100% |

No signer appears in more than one split. Every retained class is represented in all three splits. Signer independence is conditional on consistent and accurate signer annotation. The total class count is the shared vocabulary size, not the sum of the three class counts.

The partition was selected using 100 multistart local-search runs with seed `20261009` and up to 80 single-signer improvement steps per initialization, prioritizing retained-class coverage while constraining training share to 55–65% and validation/test shares to 15–25%. Global optimality is not certified. The search does not use test recognition scores.

The remaining **3,412 videos across 735 classes** are excluded from this benchmark and preserved in the full manifest. Of these classes, 489 have fewer than three signers; another 246 are excluded under the selected joint coverage/ratio solution. The latter are not proven impossible under every assignment.

## Dataset access and metadata

Project repository: [https://github.com/dinhthang12t/VSL-ISLR-Benchmark](https://github.com/dinhthang12t/VSL-ISLR-Benchmark).

Dataset and predefined split access: `[TO COMPLETE: exact file paths in the repository or a separate dataset URL/DOI]`. The repository URL identifies the project; it does not by itself confirm that video files or every split manifest have been uploaded.

Source-video access instructions: `[TO COMPLETE: verified provenance and access information]`.

Metadata fields:

| Column | Description |
|---|---|
| `ID_video` | Video identifier. |
| `Meaning` | Vietnamese sign-class label. |
| `Duration (s)` | Video duration in seconds. |
| `Signer` | Annotated signer identifier. |
| `split` | Original class-stratified split: train/valid/test. |
| `split_signer_independent` | New split: train/val/test, or excluded in the full manifest. |

For the new benchmark, use **`split_signer_independent`**, not `split`. The benchmark-only file `ISLR_447classes_4252videos.csv` has no excluded rows. Metadata and splits can be hosted separately from the code; link their actual location above rather than assuming additional repository folders.

Public source videos are not automatically redistributable. See [DATA_LICENSE.md](DATA_LICENSE.md) for source rights and release conditions.

## 447-class split package

The following resources accompany the split package. Link their actual download location in the data-access section above; the table does not assume additional code directories outside `src/`.

| File | Contents and use |
|---|---|
| `ISLR_447classes_4252videos.csv` | Filtered benchmark manifest: 4,252 rows across 447 classes; use for training and evaluation. |
| `ISLR_447_all_rows.csv` | Complete 7,664-row source manifest; 3,412 videos across 735 excluded classes remain marked `excluded`. |
| `class_split_audit.csv` | Audit of all 1,182 source classes, with video/signer counts, new split counts, and retention/exclusion status. |
| `signer_assignment.csv` | All 57 source signer IDs: 50 have benchmark videos and seven are marked `not_in_benchmark`. The latter are outside this benchmark, not unidentified signers. |
| `validation_checks.csv` | Pairwise signer-overlap checks, video-ID uniqueness, and coverage of all 447 classes across the three splits. |
| `split_summary.json` | Machine-readable split statistics and search settings. |
| `Reviewer_responses_VI.md` / `Reviewer_responses_EN.md` | Revision working documents addressing six comments from Reviewer 1 and ten from Reviewer 2; distinguish completed split preparation from pending experiments. These are supplementary revision materials, not benchmark results. |

The source `README_447.md` stated 51 retained signers and six excluded signers. Those figures have been corrected here against `signer_assignment.csv`: **25 + 13 + 12 = 50 retained signers**, and **57 − 50 = 7 signers outside the benchmark**.

### Copying videos into split folders

When adapting a copy script written for the original CSV, read `row["split_signer_independent"]` instead of `row["split"]`. Copy videos into `<output>/<train|val|test>/<Meaning>/`, preserving the correspondence with `ID_video`. The benchmark-only CSV needs no exclusion filter. When using the complete manifest, skip rows marked `excluded`.

A signer must keep the same ID across different clothing, backgrounds or recording sessions. Such changes do not make the person a new signer.

## Installation

Clone the repository:

```bash
git clone https://github.com/dinhthang12t/VSL-ISLR-Benchmark.git
cd VSL-ISLR-Benchmark
```

### Environment reported in the manuscript

The following information is taken from Sections 3.2 and 4.2 of the supplied manuscript. It describes the reported original experiments; it must be checked against the implementation and logs before being used as the configuration of a new run.

| Item | Reported setting |
|---|---|
| Hardware | A single NVIDIA A100 GPU; GPU memory capacity is not specified. |
| Training precision | Mixed precision. |
| Runtime measurement precision | FP16 inference. |
| Python version | Not specified in the manuscript; record the version from the actual environment. |
| Deep-learning framework/version | Not specified in the manuscript; record the actual framework and version. |
| CUDA, cuDNN and dependency versions | Not specified in the manuscript; export them from the actual environment. |

Installation instructions: `[TO COMPLETE: tested commands and dependency versions for the implementation under src/]`.

The manuscript does not provide a dependency installation command. Do not infer package versions, an A100 memory capacity, or a CUDA version from the GPU model.

## Running the experiments

### Training configuration reported in Section 4.2

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Backbone learning rate | `1e-4` |
| Classification-head learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Label smoothing | `0.05` |
| Reported batch size | `16`; effective batch size requires the accumulation factor. |
| Maximum epochs | `300` |
| Early-stopping patience | `10` epochs; monitored metric and minimum improvement are not specified. |
| Gradient accumulation | Used; accumulation factor is not specified. |
| Initialization | Pretrained weights; exact checkpoint sources are not specified. |
| Fine-tuning | Early layers frozen; deeper layers and classification heads optimized. Exact frozen layers are not specified. |
| Random seeds shown in the original multi-seed table | `42`, `123`, `2024`, for the models included in that table. |

### Model inputs and preprocessing

| Pipeline | Frames | Spatial input / representation |
|---|---:|---|
| RGB models | 64 | Generally `224 × 224`; C3D uses `112 × 112`. |
| Pose-based models | 64 | 2D keypoints; training sequences may be cropped/dropped and padded within each batch. |

RGB inputs use ImageNet normalization as reported in the manuscript. Section 3.2 describes uniform (`iter`), center-focused (`mid`), and mixed (`mix`) sampling. Section 4.2 uses different sampling terminology; reconcile these names with the implementation and identify the strategy used for each reported result before release.

These original input configurations differ across architectures. They must not be described as a matched-frame/matched-resolution controlled comparison. If new controlled experiments use other inputs, report those settings separately.

### Implementation entry points

All implementation files are organized under `src/`. The manuscript does not specify their names or command-line arguments, so the following fields require the actual source code:

- Training: `[TO COMPLETE: actual entry-point filename under src/ and tested command]`.
- Evaluation: `[TO COMPLETE: actual entry-point filename under src/ and tested command]`.
- Configuration: `[TO COMPLETE: actual source location or command-line options]`.

For reproducibility, also record the exact backbone variants and checkpoint sources, frozen layers, gradient accumulation factor, early-stopping metric, and GRU hidden size/layers/directionality/dropout. These details cannot be recovered reliably from the manuscript alone.

Train each protocol separately and select checkpoints using validation only. Do not use a checkpoint fitted to the original VSL split to establish independence on the new split. Use one consistent class-to-index mapping across train/validation/test within each protocol.

For a local copy of the new manifest:

```python
import pandas as pd

df = pd.read_csv("ISLR_447classes_4252videos.csv")
train = df[df["split_signer_independent"] == "train"]
val = df[df["split_signer_independent"] == "val"]
test = df[df["split_signer_independent"] == "test"]
class_to_idx = {label: i for i, label in enumerate(sorted(df["Meaning"].unique()))}

assert not df["ID_video"].duplicated().any()
assert df.groupby("Signer")["split_signer_independent"].nunique().max() == 1
assert df.groupby("Meaning")["split_signer_independent"].nunique().min() == 3
```

## Results and limitations

Result access and reproduction instructions for manuscript tables/figures: `[TO COMPLETE: actual links, commands and configuration references]`.

Report the protocols separately, including Top-1/Top-5/Top-10 accuracy, Macro-F1, per-seed scores, and mean ± sample standard deviation. Document the mapping from each table to its experimental settings.

The two benchmarks differ in vocabulary and video composition; accuracy differences cannot be attributed solely to signer overlap. The new training split averages 5.24 videos per class, and some classes have only one training video. Report uncertainty and avoid strong per-class claims from single test examples. The split does not enforce a minimum of three training videos or two training signers per class. Define head/medium/tail groups from training frequencies and report the corresponding support counts.

Identity consistency, duplicate video content and labels have not been verified directly from source recordings during split construction. In particular, the label for `Y0668` follows the supplied corrected metadata rather than an independent video-content check. The full-corpus claim of over 80 signers requires separate verification; it must not be inferred from the 57 IDs in the supplied metadata. Completing the split alone does not satisfy the remaining experimental requests from reviewers.

## Citation and contact

Paper citation: `[TO COMPLETE: verified authors, publication details and DOI]`.

Repository version: `[TO COMPLETE: release tag or commit]`.

Maintainer/contact: `[TO COMPLETE: verified name and email]`.

Report reproducibility or annotation issues through [GitHub Issues](https://github.com/dinhthang12t/VSL-ISLR-Benchmark/issues), including the affected video IDs and, for implementation issues, the repository commit and environment details. Changes to published manifests should receive a new version with a documented correction history.
