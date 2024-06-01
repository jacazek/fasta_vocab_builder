#!/bin/bash
#SBATCH --exclusive
export TQDM_DISABLE=1
rocm-python src/build_vocabulary.py --kmer_size 6 --stride 3

