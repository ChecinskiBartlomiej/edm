#!/usr/bin/env python3
"""
Reproduce Table 2 from EDM paper using STREAMING FID (no saving images, no 50k in RAM).
Works with torchrun (multi-GPU) and also with single GPU.

Pipeline (per run):
seeds -> split across ranks -> for each batch: generate -> inception -> accumulate sums
-> all_reduce sums -> compute mu/sigma -> rank0 computes FID vs reference stats
"""

import os
import json
from pathlib import Path
import sys

import numpy as np
import torch

# Make local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_utils import distributed as dist
from generate import load_edm_network, split_seeds_across_ranks, generate_images_for_seeds_batch
from fid import load_inception_network, calculate_fid_from_inception_stats


# -----------------------------
# Your Table2 constants (unchanged)

TABLE2_REFERENCE = {
    "CIFAR-10 Conditional": {
        "VP": {"A": {"FID": 2.48, "NFE": 35}, "F": {"FID": 1.79, "NFE": 35}},
        "VE": {"A": {"FID": 3.11, "NFE": 35}, "F": {"FID": 1.79, "NFE": 35}},
    },
    "CIFAR-10 Unconditional": {
        "VP": {"A": {"FID": 3.01, "NFE": 35}, "F": {"FID": 1.97, "NFE": 35}},
        "VE": {"A": {"FID": 3.77, "NFE": 35}, "F": {"FID": 1.98, "NFE": 35}},
    },
    "FFHQ 64x64": {
        "VP": {"A": {"FID": 3.39, "NFE": 79}, "F": {"FID": 2.39, "NFE": 79}},
        "VE": {"A": {"FID": 25.95, "NFE": 79}, "F": {"FID": 2.53, "NFE": 79}},
    },
    "AFHQv2 64x64": {
        "VP": {"A": {"FID": 2.58, "NFE": 79}, "F": {"FID": 1.96, "NFE": 79}},
        "VE": {"A": {"FID": 18.52, "NFE": 79}, "F": {"FID": 2.16, "NFE": 79}},
    },
}

FID_REFS = {
    "CIFAR-10": "fid_refs/cifar10-32x32.npz",
    "FFHQ": "fid_refs/ffhq-64x64.npz",
    "AFHQv2": "fid_refs/afhqv2-64x64.npz",
}

BASELINE_MODELS = {
    ("CIFAR-10 Conditional", "VP"): "config_a_pretrained_models/baseline-cifar10-32x32-cond-vp.pkl",
    ("CIFAR-10 Conditional", "VE"): "config_a_pretrained_models/baseline-cifar10-32x32-cond-ve.pkl",
    ("CIFAR-10 Unconditional", "VP"): "config_a_pretrained_models/baseline-cifar10-32x32-uncond-vp.pkl",
    ("CIFAR-10 Unconditional", "VE"): "config_a_pretrained_models/baseline-cifar10-32x32-uncond-ve.pkl",
    ("FFHQ 64x64", "VP"): "config_a_pretrained_models/baseline-ffhq-64x64-uncond-vp.pkl",
    ("FFHQ 64x64", "VE"): "config_a_pretrained_models/baseline-ffhq-64x64-uncond-ve.pkl",
    ("AFHQv2 64x64", "VP"): "config_a_pretrained_models/baseline-afhqv2-64x64-uncond-vp.pkl",
    ("AFHQv2 64x64", "VE"): "config_a_pretrained_models/baseline-afhqv2-64x64-uncond-ve.pkl",
}

CONFIG_F_MODELS = {
    ("CIFAR-10 Conditional", "VP"): "config_f_pretrained_models/edm-cifar10-32x32-cond-vp.pkl",
    ("CIFAR-10 Conditional", "VE"): "config_f_pretrained_models/edm-cifar10-32x32-cond-ve.pkl",
    ("CIFAR-10 Unconditional", "VP"): "config_f_pretrained_models/edm-cifar10-32x32-uncond-vp.pkl",
    ("CIFAR-10 Unconditional", "VE"): "config_f_pretrained_models/edm-cifar10-32x32-uncond-ve.pkl",
    ("FFHQ 64x64", "VP"): "config_f_pretrained_models/edm-ffhq-64x64-uncond-vp.pkl",
    ("FFHQ 64x64", "VE"): "config_f_pretrained_models/edm-ffhq-64x64-uncond-ve.pkl",
    ("AFHQv2 64x64", "VP"): "config_f_pretrained_models/edm-afhqv2-64x64-uncond-vp.pkl",
    ("AFHQv2 64x64", "VE"): "config_f_pretrained_models/edm-afhqv2-64x64-uncond-ve.pkl",
}

# Path to local inception detector
INCEPTION_PATH = "inception_v3_2/inception-2015-12-05.pkl"

# -----------------------------
# Helpers

def get_device_from_torchrun():
    """Pick correct GPU for this process. Works with torchrun; on 1 GPU LOCAL_RANK may be absent."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def load_fid_ref_npz(ref_path):
    """Load reference mu/sigma (rank0 only)."""
    with open(ref_path, "rb") as f:
        ref = dict(np.load(f))
    return ref["mu"], ref["sigma"]


def find_optimal_steps(target_nfe, use_heun=True):
    """Paper uses Heun: NFE = 2*num_steps - 1."""
    if use_heun:
        return (target_nfe + 1) // 2
    return target_nfe


def pick_fid_ref_for_dataset(dataset_name):
    if "CIFAR-10" in dataset_name:
        return FID_REFS["CIFAR-10"]
    if "FFHQ" in dataset_name:
        return FID_REFS["FFHQ"]
    if "AFHQv2" in dataset_name:
        return FID_REFS["AFHQv2"]
    raise ValueError(f"Unknown dataset name: {dataset_name}")


@torch.no_grad()
def compute_fid_streaming_one_run(
    device,
    inception_net,
    edm_net,
    fid_ref_path,
    seeds,
    max_batch_size=64,
    num_steps=18,
):
    """
    One run: streaming computation of inception stats across ranks, then FID on rank0.
    No saving images, no storing 50k images.

    Returns:
        fid (float) on rank0, None on other ranks.
    """
    # Load generator once per run per process
    
    feature_dim = 2048
    detector_kwargs = dict(return_features=True)

    sum_feat = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sum_outer = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    count = torch.zeros([], dtype=torch.long, device=device)

    # Each rank gets its own seed batches
    rank_batches = split_seeds_across_ranks(seeds, max_batch_size=max_batch_size)

    for batch_seeds in rank_batches:
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # 1) generate images in [-1,1], [B,C,H,W]
        images = generate_images_for_seeds_batch(
            device,
            edm_net,
            batch_seeds=batch_seeds,
            class_idx=None,           # random labels for conditional models
            num_steps=num_steps,
        )

        # 2) ensure 3 channels + scale to [0,255] for inception
        images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        
        # 3) features [B,2048]
        feats = inception_net(images, **detector_kwargs).to(torch.float64)

        # 4) accumulate sums
        sum_feat += feats.sum(0)
        sum_outer += feats.T @ feats
        count += feats.shape[0]

    # Global sums across ranks
    torch.distributed.all_reduce(sum_feat)
    torch.distributed.all_reduce(sum_outer)
    torch.distributed.all_reduce(count)

    N = int(count.item())
    if N < 2:
        raise RuntimeError(f"Need at least 2 images, got {N}")
    if N != len(seeds):
        dist.print0(f"Warning: expected {len(seeds)} images, got {N}")

    # mu, sigma (unbiased covariance)
    mu_t = sum_feat / N
    sigma_t = (sum_outer - N * mu_t.ger(mu_t)) / (N - 1)

    fid_val = None
    if dist.get_rank() == 0:
        mu_ref, sigma_ref = load_fid_ref_npz(fid_ref_path)
        fid_val = calculate_fid_from_inception_stats(
            mu=mu_t.cpu().numpy(),
            sigma=sigma_t.cpu().numpy(),
            mu_ref=mu_ref,
            sigma_ref=sigma_ref,
        )

    torch.distributed.barrier()
    return fid_val


def reproduce_config_streaming(device, inception_net, edm_net, dataset, formulation, config_name, num_images, num_runs):
    """
    Reproduce one config using streaming stats (multi-GPU correct).
    Only rank0 prints and returns dict; other ranks return None.
    """
    ref = TABLE2_REFERENCE[dataset][formulation][config_name]
    target_nfe = ref["NFE"]
    expected_fid = ref["FID"]

    fid_ref = pick_fid_ref_for_dataset(dataset)
    num_steps = find_optimal_steps(target_nfe, use_heun=True)

    if dist.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"Config {config_name}: {dataset} - {formulation} (STREAMING, no 50k RAM)")
        print(f"{'='*80}")
        print(f"Target NFE: {target_nfe} -> num_steps: {num_steps}")
        print(f"FID ref: {fid_ref}")
        print(f"Runs: {num_runs}")

    fid_scores = []

    for run_idx in range(num_runs):
        seed_start = run_idx * num_images
        seeds = list(range(seed_start, seed_start + num_images))

        if dist.get_rank() == 0:
            print(f"\n--- Run {run_idx + 1}/{num_runs} (seeds {seed_start}-{seed_start + (num_images-1)}) ---")

        fid_val = compute_fid_streaming_one_run(
            device,
            inception_net,
            edm_net,
            fid_ref_path=fid_ref,
            seeds=seeds,
            max_batch_size=64,
            num_steps=num_steps,
        )

        if dist.get_rank() == 0:
            fid_scores.append(float(fid_val))
            print(f"FID run {run_idx + 1}: {fid_val:.4f}")

    if dist.get_rank() != 0:
        return None

    min_fid = min(fid_scores)
    print(f"\n{'='*80}")
    print("Final Results:")
    print(f"  All FID scores: {[f'{f:.4f}' for f in fid_scores]}")
    print(f"  Minimum FID:    {min_fid:.4f} (reported value)")
    print(f"  Expected FID:   {expected_fid:.4f} (from paper)")
    print(f"  Difference:     {abs(min_fid - expected_fid):.4f}")
    print(f"  NFE:            {target_nfe}")
    print(f"  num_steps:      {num_steps}")
    print(f"{'='*80}")

    return {
        "dataset": dataset,
        "formulation": formulation,
        "config": config_name,
        "expected_fid": expected_fid,
        "all_fids": fid_scores,
        "min_fid": min_fid,
        "actual_fid": min_fid,
        "nfe": target_nfe,
        "num_steps": num_steps,
        "fid_ref": fid_ref,
    }


def reproduce_all_configs_streaming(device, inception_net, output_base, num_images, num_runs):
    """Run all configs. Must be called by all ranks (same control flow)."""
    results = {"config_a": [], "config_f": []}

    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("REPRODUCING CONFIG A (BASELINE) - STREAMING")
        print("="*80)

    for dataset in TABLE2_REFERENCE:
        for formulation in TABLE2_REFERENCE[dataset]:
            key = (dataset, formulation)
            edm_net = load_edm_network(BASELINE_MODELS[key], device)
            r = reproduce_config_streaming(
                device,
                inception_net,
                edm_net,
                dataset=dataset,
                formulation=formulation,
                config_name="A",
                num_images=num_images,
                num_runs=num_runs,
            )
            if dist.get_rank() == 0 and r is not None:
                results["config_a"].append(r)

    if dist.get_rank() == 0:
        print("\n" + "="*80)
        print("REPRODUCING CONFIG F (ALL IMPROVEMENTS) - STREAMING")
        print("="*80)

    for dataset in TABLE2_REFERENCE:
        for formulation in TABLE2_REFERENCE[dataset]:
            key = (dataset, formulation)
            edm_net = load_edm_network(CONFIG_F_MODELS[key], device)
            r = reproduce_config_streaming(
                device,
                inception_net,
                edm_net,
                dataset=dataset,
                formulation=formulation,
                config_name="F",
                num_images=num_images,
                num_runs=num_runs,
            )
            if dist.get_rank() == 0 and r is not None:
                results["config_f"].append(r)

    # Save on rank0
    if dist.get_rank() == 0:
        out = Path(output_base)
        out.mkdir(parents=True, exist_ok=True)

        config_a_file = out / "config_a_results_streaming.json"
        config_f_file = out / "config_f_results_streaming.json"

        with open(config_a_file, "w") as f:
            json.dump(results["config_a"], f, indent=2)
        with open(config_f_file, "w") as f:
            json.dump(results["config_f"], f, indent=2)

        print(f"\nSaved:\n  {config_a_file}\n  {config_f_file}")

        # Summary tables
        print("\nConfig A Summary Table:")
        print(f"{'Dataset':<30} {'Form':<5} {'Expected':<10} {'Min FID':<10} {'Diff':<10}")
        print("-" * 80)
        for r in results["config_a"]:
            print(f"{r['dataset']:<30} {r['formulation']:<5} {r['expected_fid']:<10.2f} {r['min_fid']:<10.2f} {abs(r['min_fid']-r['expected_fid']):<10.2f}")

        print("\nConfig F Summary Table:")
        print(f"{'Dataset':<30} {'Form':<5} {'Expected':<10} {'Min FID':<10} {'Diff':<10}")
        print("-" * 80)
        for r in results["config_f"]:
            print(f"{r['dataset']:<30} {r['formulation']:<5} {r['expected_fid']:<10.2f} {r['min_fid']:<10.2f} {abs(r['min_fid']-r['expected_fid']):<10.2f}")

    torch.distributed.barrier()


def main():
    # Init DDP once
    dist.init()

    device = get_device_from_torchrun()
    inception_net = load_inception_network(INCEPTION_PATH, device)

    dist.print0("=" * 80)
    dist.print0("Table 2 Reproduction Script (STREAMING VERSION)")
    dist.print0("=" * 80)
    dist.print0(f"DDP world_size={dist.get_world_size()} rank={dist.get_rank()} device={device}")
    dist.print0("Mode: STREAMING (no disk I/O for images, no 50k images in RAM)")

    output_dir = "results"
    num_runs = 3
    num_images = 50000

    reproduce_all_configs_streaming(device, inception_net, output_dir, num_images=num_images, num_runs=num_runs)

    dist.print0("Done.")


if __name__ == "__main__":
    main()