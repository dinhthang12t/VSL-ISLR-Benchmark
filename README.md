# VSL-ISLR Benchmark

Resources for **A Comparative Study of Temporal Modeling for Large-Scale Vietnamese Isolated Sign Language Recognition**.

> Publication draft: replace all `[TO COMPLETE: ...]` fields and verify the release contents before publishing. This document does not establish that code, videos, checkpoints, or new experimental results have already been released.

## Overview

This project compares temporal modeling approaches for Vietnamese isolated sign language recognition, including 2D CNN features with mean pooling or GRU aggregation, 3D CNNs, video transformers, and pose-based temporal graph models.

Two evaluation protocols are distinguished:

1. **Class-stratified protocol with signer overlap:** the original evaluation setting. Signers may occur in multiple splits; results do not establish generalization to unseen signers.
2. **Signer-independent 447-class protocol:** an additional benchmark with mutually disjoint training, validation, and test signer sets.

## Dataset versions

| Resource | Videos | Classes | Signers | Description |
|---|---:|---:|---:|---|
| Full corpus described in the manuscript | 10,515 | 3,659 | To be verified | Full-corpus counts must be reconciled against the final release manifest. |
| Current signer-annotated metadata | 7,664 | 1,182 | 57 | Corrected metadata supplied for split construction. |
| Signer-independent benchmark | 4,252 | 447 | 50 | Filtered benchmark used for unseen-signer evaluation. |

The earlier manuscript reports 7,665 benchmark videos. The current metadata contains 7,664 unique video IDs after a duplicate-ID correction. Verify the corrected annotation against the source video before freezing the release. Do not silently combine results generated from different metadata versions.

### Signer-independent split

| Split | Videos | Signers | Classes | Video share |
|---|---:|---:|---:|---:|
| Training | 2,344 | 25 | 447 | 55.13% |
| Validation | 1,021 | 13 | 447 | 24.01% |
| Test | 887 | 12 | 447 | 20.86% |
| Total | 4,252 | 50 | 447 | 100% |

Every retained class appears in all three splits. Class counts are shared vocabulary counts and must not be added across splits. Signer separation relies on the accuracy of the annotated identities.

The split was selected using multistart local search with seed `20261009`, 100 initializations, and up to 80 improvement steps per initialization. The search favors class coverage while constraining training video share to 55–65% and validation/test shares to 15–25%. It does not use recognition accuracy to select the partition. Global optimality is not certified.

Of 1,182 source classes, 489 contain fewer than three signers. A further 246 classes are excluded under the selected joint coverage/ratio solution; this does not prove they are impossible under every alternative assignment. The full manifest preserves all excluded rows. Training averages 5.24 videos per class, but some classes have only one training video.

## Metadata and manifests

| Column | Meaning |
|---|---|
| `ID_video` | Video identifier; match to the corresponding local filename. |
| `Meaning` | Vietnamese sign-class label. |
| `Duration (s)` | Clip duration in seconds. |
| `Signer` | Annotated signer identifier; treat as an opaque ID, not a demographic attribute. |
| `split` | Original class-stratified assignment (`train`, `valid`, `test`). |
| `split_signer_independent` | New assignment (`train`, `val`, `test`, or `excluded` in the full manifest). |

Use `split_signer_independent` for the 447-class protocol. The benchmark-only CSV contains no `excluded` rows. If exporting separate train/val/test files, derive them directly from the frozen manifest.

Suggested release layout; include a path below only after adding the actual resource:

```text
metadata/
  dataset_metadata.csv
splits/
  class_stratified/
  signer_independent_447/
    ISLR_447classes_4252videos.csv
    ISLR_447_all_rows.csv
    class_split_audit.csv
    signer_assignment.csv
    validation_checks.csv
configs/
src/
scripts/
results/
docs/
  reproduction.md
  model_details.md
README.md
DATA_LICENSE.md
LICENSE
requirements.txt
```

## Data access

- Metadata and frozen splits: `[TO COMPLETE: repository path and versioned URL/DOI]`.
- Source provenance and access instructions: `[TO COMPLETE: source registry, original URLs, clip boundaries, and applicable permissions]`.
- Redistributable video files, if authorized: `[TO COMPLETE: scope and download location; otherwise state that videos are not redistributed]`.

The manuscript identifies YouTube and QIPEDC as source collections. Public accessibility alone does not establish permission to redistribute or relicense their videos. See [DATA_LICENSE.md](DATA_LICENSE.md). Internal video identifiers alone may not enable independent acquisition; supply source mappings and temporal boundaries where permitted.

## Setup and reproduction

Implementation release: `[TO COMPLETE: code URL, release tag/commit]`.

Before publishing, document the tested Python, framework, CUDA and dependency versions in `requirements.txt` and `docs/reproduction.md`. Include tested commands with actual script arguments; executable training commands are intentionally not invented in this draft.

For every reported model, provide:

- Exact backbone variant, initialization/pretraining dataset and checkpoint provenance.
- Input frame count, image resolution, temporal sampling, normalization and augmentation.
- Frozen layers, optimizer, learning rates, regularization, batch size and stopping rule.
- GRU configuration and tuning budget where applicable.
- Seeds, checkpoint-selection rule, hardware and evaluation settings.
- Mapping from each manuscript table/figure to its configuration and command.

Train each protocol separately. Use validation only for tuning and checkpoint selection. Do not initialize the new evaluation from a checkpoint already fitted to videos/signers now in its test set. Use one consistent label mapping across all splits of a protocol; the two protocols have different label spaces.

### Read and check the 447-class split

```python
import pandas as pd

df = pd.read_csv(
    "splits/signer_independent_447/ISLR_447classes_4252videos.csv"
)
parts = {
    s: df[df["split_signer_independent"] == s]
    for s in ("train", "val", "test")
}
assert len(df) == 4252 and df["Meaning"].nunique() == 447
assert not df["ID_video"].duplicated().any()
assert df.groupby("Signer")["split_signer_independent"].nunique().max() == 1
assert df.groupby("Meaning")["split_signer_independent"].nunique().min() == 3
class_to_idx = {label: i for i, label in enumerate(sorted(df["Meaning"].unique()))}
```

## Results and interpretation

Report the two protocols in separate tables. Include Top-1, Top-5, Top-10 and Macro-F1, per-seed scores, mean ± sample standard deviation, and the exact run/configuration identifiers. Define head/medium/tail groups using training frequencies. Publish per-video predictions where permitted.

Result files and manuscript table mapping: `[TO COMPLETE: verified paths]`.

Differences between the original and 447-class benchmarks cannot be attributed solely to signer overlap because the vocabulary and video composition also change. Qualitative Grad-CAM examples do not by themselves prove temporal reasoning or background invariance.

## Licensing and citation

Data rights and access conditions are described in [DATA_LICENSE.md](DATA_LICENSE.md). Code licensing must be specified separately in `LICENSE`; do not assume it covers videos or pretrained weights.

Citation: `[TO COMPLETE: verified authors, publication details, DOI/arXiv URL, and BibTeX]`.

## Contact and corrections

Maintainer/contact: `[TO COMPLETE: verified name and email]`.

For annotation errors, provide the video ID, affected field and supporting information. Freeze revised manifests under a new version, document changes, and identify which version was used for each result. Rights holders may contact the maintainer about source attribution or access concerns.
