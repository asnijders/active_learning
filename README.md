# Investigating Multi-source Active Learning for Natural Language Inference.

This repository is organized as follows:

* A Conda environment can be found in `resources`\ `environment`.
* Datasets can be downloaded by running `scripts`\ `download.sh`. Note: This script does _not_ download WANLI data as it was not yet publicly available at time of submission.
* SLURM job scripts are located in `job_scripts`.
* Source code is located in `src` and contains implementations for most of the necessary logic for loading and building datasets, models, active learning strategies, analysis metrics and related utilities. 
* The main active learning loop is implemented in `main.py`.
 
