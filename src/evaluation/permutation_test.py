"""
Exact Paired Permutation Test with Holm-Bonferroni Correction for Table 6.
Conducts exact two-sided sign-flip permutation tests (2^N permutations for N seeds).
"""

import argparse
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd


def paired_permutation_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Computes exact two-sided paired sign-permutation test.
    Returns (mean_diff, raw_p_value).
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    assert len(a_arr) == len(b_arr), "Samples must be paired and have equal length"

    diffs = a_arr - b_arr
    n = len(diffs)
    observed_diff = float(np.mean(diffs))
    obs_t = abs(observed_diff)

    # Enumerate all 2^N sign flips
    total_permutations = 2 ** n
    count_as_or_more_extreme = 0

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        perm_stat = abs(np.mean(diffs * np.array(signs)))
        if perm_stat >= obs_t - 1e-12:
            count_as_or_more_extreme += 1

    raw_p = count_as_or_more_extreme / total_permutations
    return observed_diff, raw_p


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction for multiple comparisons.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        val = p * multiplier
        cum_max = max(cum_max, val)
        adjusted[orig_idx] = min(1.0, cum_max)

    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Exact Paired Permutation Test (Table 6)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save summary CSV")
    args = parser.parse_args()

    # Paired runs across 7 seeds: [42, 123, 2024, 7, 19, 88, 4096]
    # Example runs matching Table 5/6:
    swin3d_scores = [60.51, 60.35, 60.42, 60.48, 60.29, 60.55, 60.38]
    x3d_scores    = [54.70, 55.80, 56.40, 55.20, 57.10, 54.90, 56.10]

    vitpose_tgcn_scores = [49.06, 48.55, 47.80, 50.12, 48.91, 46.85, 50.45]
    resnet50_mean_scores = [46.89, 47.12, 47.35, 46.95, 47.41, 47.28, 47.19]

    comparisons = [
        ("Swin3D - X3D", swin3d_scores, x3d_scores),
        ("ViTPose + TGCN - ResNet-50 + Mean", vitpose_tgcn_scores, resnet50_mean_scores),
    ]

    results = []
    raw_p_list = []

    for name, s_a, s_b in comparisons:
        mean_diff, raw_p = paired_permutation_test(s_a, s_b)
        results.append({
            "Comparison (A - B)": name,
            "Mean difference (pp)": f"{mean_diff:+.2f}",
            "raw_p": raw_p,
        })
        raw_p_list.append(raw_p)

    # Apply Holm correction
    holm_p_list = holm_bonferroni(raw_p_list)

    for i in range(len(results)):
        results[i]["Raw p-value"] = f"{results[i]['raw_p']:.4f}"
        results[i]["Holm-adjusted p-value"] = f"{holm_p_list[i]:.4f}"
        del results[i]["raw_p"]

    df_out = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("TABLE 6: Statistical Significance (Exact Paired Permutation Test)")
    print("=" * 75)
    print(df_out.to_string(index=False))
    print("=" * 75)

    if args.output_csv:
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
