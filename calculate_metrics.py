# Copyright (c) 2026, Signal Processing Group. University of Hamburg. All rights reserved.
#
# Code adapted from https://github.com/NVlabs/edm2
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Compute intrusive (reference-based) speech enhancement metrics."""

import os
import csv
from glob import glob
from os.path import join
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from soundfile import read

from pesq import pesq


SR = 16000


def sisdr(reference, estimation):
    """
    Code from: https://github.com/fgnt/pb_bss/blob/master/pb_bss/evaluation/module_si_sdr.py
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    assert reference.dtype == np.float64, reference.dtype
    assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def _load_and_match(proc_file: str, target_file: str):
    proc_audio, proc_sr = read(proc_file)
    target_audio, target_sr = read(target_file)

    if proc_sr != SR or target_sr != SR:
        raise ValueError(
            f"Sample rate mismatch: {proc_file}={proc_sr}, {target_file}={target_sr} (expected {SR})"
        )

    # Ensure 1D
    proc_audio = np.asarray(proc_audio).squeeze()
    target_audio = np.asarray(target_audio).squeeze()

    # Trim to same length
    min_len = min(len(proc_audio), len(target_audio))
    proc_audio = proc_audio[:min_len]
    target_audio = target_audio[:min_len]

    return proc_audio, target_audio


def process_file(proc_file: str, proc_dir: str, target_file: str):
    proc_audio, target_audio = _load_and_match(proc_file, target_file)

    filename = proc_file.replace(proc_dir, "").lstrip("/")

    return {
        "filename": filename,
        "sisdr": float(sisdr(target_audio, proc_audio)),
        "pesq": float(pesq(SR, target_audio, proc_audio, "wb")),
    }


def main(args):
    proc_files = sorted(glob(join(args.proc_dir, "**", "*.wav"), recursive=True))
    target_files = sorted(glob(join(args.target_dir, "**", "*.wav"), recursive=True))

    if args.debug:
        proc_files = proc_files[:2]
        target_files = target_files[:2]

    if len(proc_files) != len(target_files):
        raise RuntimeError(
            f"Number of files do not match: proc_files={len(proc_files)} target_files={len(target_files)}"
        )

    print(f"Processing metrics for {len(proc_files)} files...")

    rows = []
    for proc_file, target_file in tqdm(list(zip(proc_files, target_files)), total=len(proc_files)):
        rows.append(process_file(proc_file, args.proc_dir, target_file))

    # Means (ignore NaNs just in case)
    sisdr_mean = float(np.nanmean([r["sisdr"] for r in rows])) if rows else float("nan")
    pesq_mean = float(np.nanmean([r["pesq"] for r in rows])) if rows else float("nan")

    print("\nMean metrics:")
    print(f"  SI-SDR : {sisdr_mean:.2f}")
    print(f"  PESQ   : {pesq_mean:.2f}")

    # Save CSV
    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = join(args.results_dir, f"{args.name}.csv")

    fieldnames = ["filename", "sisdr", "pesq"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    """
    Usage:

    python calculate_metrics.py \
        --proc_dir /path/to/enhanced_wavs \
        --target_dir /path/to/clean_wavs \
        --results_dir /path/to/results \
        --name run_name
    """
    parser = ArgumentParser()
    parser.add_argument("--proc_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
