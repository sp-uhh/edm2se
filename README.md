# EDM2SE

**A Magnitude-Preserving Network Architecture for Diffusion-Based Speech
Enhancement**

This repository contains inference code for **EDM2SE**, a
diffusion-based speech enhancement model with a magnitude-preserving
network architecture.

The codebase is adapted from the official EDM2 implementation: https://github.com/NVlabs/edm2

------------------------------------------------------------------------

## Setup

We recommend using a dedicated conda environment.

``` bash
conda create --name edm2se python=3.10
conda activate edm2se
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Inference

### Download Pretrained Checkpoint

Download the checkpoint for EDM2SE trained on VoiceBank-DEMAND
(VB-DMD):

https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2026-edm2se/checkpoints/edm2se_vbdmd.ckpt

------------------------------------------------------------------------

### Run Enhancement

``` bash
python generate.py
	--net /path/to/checkpoint.ckpt
	--test_dir /path/to/noisy_dir
	--outdir /path/to/enhanced_dir
```

**Arguments:** 

- `--net` Path to the pretrained EDM2SE checkpoint\
- `--test_dir` Directory containing noisy input WAV files\
- `--outdir` Output directory for enhanced WAV files

------------------------------------------------------------------------

## Reference-Based Metrics

To compute PESQ and SI-SDR between enhanced and clean reference
signals:

``` bash
python calculate_metrics.py     
	--proc_dir /path/to/enhanced_wavs
	--target_dir /path/to/clean_wavs
	--results_dir /path/to/results
	--name run_name
```

**Arguments:** 

- `--proc_dir` Directory containing enhanced WAV
files\
- `--target_dir` Directory containing clean reference WAV files\
- `--results_dir` Output directory for CSV metric files\
- `--name` Run identifier used for naming result files

The script saves a CSV file and prints mean scores to the terminal.

------------------------------------------------------------------------

## Release Log

-   **01/26/2026** --- Initial inference code release

------------------------------------------------------------------------

## Training Code

Training code and configuration files will be released **soon**.

------------------------------------------------------------------------

## Citation

If you use this code or pretrained model in your work, please cite our
ICASSP 2026 paper:

> *EDM2SE: A Magnitude-Preserving Network Architecture for Diffusion-Based Speech Enhancement.*
> Julius Richter, Danilo de Oliveira, Timo Gerkmann.
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

------------------------------------------------------------------------
