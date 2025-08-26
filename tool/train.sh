#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
eval "conda activate py39"
PYTHON=/home/sx-zhang/anaconda3/envs/py39/bin/python

TRAIN_CODE=train.py

dataset=$1
exp_name=$2
exp_dir=/home/work/sx-zhang/GaitCloud-master/exp/${dataset}/${exp_name}
config=/home/work/sx-zhang/GaitCloud-master/config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${exp_dir}
cp tool/train.sh tool/${TRAIN_CODE} ${config} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")
#CUDA_LAUNCH_BLOCKING=1 
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  2>&1 | tee ${exp_dir}/train-$now.log
