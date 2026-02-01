# EDM2SE

This repository contains code for **EDM2SE**, a diffusion-based speech enhancement model with a magnitude-preserving network architecture.

This code accompanies the paper:

[Do We Need EMA for Diffusion-Based Speech Enhancement? Toward a Magnitude-Preserving Network Architecture](https://arxiv.org/abs/2505.05216)

If you use this code or build upon it in your work, please see the [Citation](#citation) section.

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

Download the checkpoint for EDM2SE trained on VoiceBank-DEMAND (VB-DMD):

https://www2.informatik.uni-hamburg.de/sp/audio/publications/icassp2026-edm2se/checkpoints/edm2se_vbdmd.ckpt

------------------------------------------------------------------------

### Run Enhancement

``` bash
python generate.py \
	--net /path/to/checkpoint.ckpt \
	--test_dir /path/to/noisy_dir \
	--out_dir /path/to/enhanced_dir
```

**Arguments:** 

- `--net` Path to the pretrained EDM2SE checkpoint
- `--test_dir` Directory containing noisy input WAV files
- `--out_dir` Output directory for enhanced WAV files

------------------------------------------------------------------------

## Reference-Based Metrics

To compute PESQ and SI-SDR between enhanced and clean reference signals:

``` bash
python calculate_metrics.py \
	--proc_dir /path/to/enhanced_wavs \
	--target_dir /path/to/clean_wavs \
	--results_dir /path/to/results \
	--name run_name
```

**Arguments:** 

- `--proc_dir` Directory containing enhanced WAV files
- `--target_dir` Directory containing clean reference WAV files
- `--results_dir` Output directory for CSV metric files
- `--name` Run identifier used for naming result files

The script saves a CSV file and prints mean scores to the terminal.

------------------------------------------------------------------------

## Release Log

-   **01/31/2026** --- Training code release 
-   **01/26/2026** --- Initial inference code release

------------------------------------------------------------------------

## Training Code

The training code trains **EDM2SE** according to the standard recipe described in the paper.

Training can be launched using distributed data parallelism. For example training EDM2SE using 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --run_dir=/path/to/run_dir \
    --data=/path/to/dataset
```

**Arguments:**

- `--nproc_per_node=2` Number of processes to launch per node (usually equal to the number of GPUs).
- `--run_dir` Directory where training outputs (checkpoints, logs, and configuration files) are saved.
- `--data`  Path to the training dataset directory.

To compute `sigma_x` and `sigma_n` for the model, run:

```bash
python training/dataset_stats.py \
    --data /path/to/data_dir
```

Make sure the project root is added to PYTHONPATH before running the script.

------------------------------------------------------------------------

## Citation

If you use this code or pretrained model in your work, please cite our ICASSP 2026 paper:

> Richter, Julius, Danilo De Oliveira, and Timo Gerkmann.
> "Do We Need EMA for Diffusion-Based Speech Enhancement? Toward a Magnitude-Preserving Network Architecture."
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026.

```bib
@inproceedings{richter2026edm2se,
  title={Do We Need {EMA} for Diffusion-Based Speech Enhancement? Toward a Magnitude-Preserving Network Architecture},
  author={Richter, Julius and de Oliveira, Danilo and Gerkmann, Timo},
  booktitle={IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year={2026}
}
```

------------------------------------------------------------------------
