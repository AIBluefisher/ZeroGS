#!/usr/bin/env bash

CUDA_IDS=$1 # {'0,1,2,...'}

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_IDS}

# Default parameters.
DATASET='blender' # [llff, mipnerf360, tanks_and_temples]
ENCODING='ace' # [ace, zero_gs]
SUFFIX=''

NUM_CMD_PARAMS=$#
if [ $NUM_CMD_PARAMS -eq 2 ]
then
    SUFFIX=$2
elif [ $NUM_CMD_PARAMS -eq 3 ]
then
    SUFFIX=$2
    DATASET=$3
elif [ $NUM_CMD_PARAMS -eq 4 ]
then
    SUFFIX=$2
    DATASET=$3
    ENCODING=$4
fi

YAML=${ENCODING}/${DATASET}'.yaml'
echo "Using yaml file: ${YAML}"

HOME_DIR=$HOME
CODE_ROOT_DIR=$HOME/'Projects/ZeroGS'

cd $CODE_ROOT_DIR

python train.py --config 'config/'${YAML} \
                --suffix $SUFFIX
