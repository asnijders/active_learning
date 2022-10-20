# Investigating Multi-source Active Learning for Natural Language Inference.

This repository is organized as follows:

* A working Conda environment with the required dependencies can be found in `resources`\ `environment`.
* Datasets can be downloaded by running `scripts`\ `download.sh`. **Note**: This script does _not_ download WANLI data as it was not yet publicly available at time of submission.
* SLURM job scripts can be found in the `job_scripts` folder.
* Source code is organized under `src` and contains implementations for most of the necessary logic for loading and building datasets, models, active learning strategies, analysis metrics and related utilities. 
* The main experiment and active learning loop is implemented in `main.py`.
 
