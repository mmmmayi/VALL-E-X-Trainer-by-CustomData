#!/bin/bash
#PBS -N train
#PBS -l select=1:ncpus=112:ngpus=8:mem=1024gb:container_engine=enroot
#PBS -l walltime=120:00:00
#PBS -q normal
#PBS -P 13003558
#PBS -j oe
#PBS -l container_image=/scratch/users/astar/ares/ma_yi/valle_custom_v3.sqsh
#PBS -l container_name=tts
#PBS -l enroot_env_file=/scratch/users/astar/ares/suns1/workspace/multimodal_trainer/scripts/a2ap_scripts/enroot_scripts/env.conf

ROOT_DIR=/home/users/astar/ares/ma_yi
SCRATCH_DIR=/scratch/users/astar/ares/ma_yi
HF_HOME=/scratch/users/astar/ares/ma_yi/hf_home
CODE_DIR=${ROOT_DIR}/code/VALL-E-X-Trainer-by-CustomData
mkdir -p /raid/local/containers/enroot-data/${PBS_JOBID}/tts/shm
chmod 777 /raid/local/containers/enroot-data/${PBS_JOBID}/tts/shm
mkdir -p $SCRATCH_DIR/tmp
export TMPDIR=$SCRATCH_DIR/tmp

mkdir -p /raid/local/containers/enroot-data/${PBS_JOBID}/tts

#change these parameters
EXP_NAME=stage-2-StartEpoch-9
PRE_NAME=stage-1
STAGE=2
START_EPOCH=10
#if your experiment config yaml is in config/experiment/suns1, then set EXP_PATH to experiment.${username}
EXP_PATH=/scratch/users/astar/ares/ma_yi/output/vallex/exp/

#If your run complains about HF permission error, then you might have to set this to 1
HF_HUB_OFFLINE=1

#Do not change these unless you are sure what you are doing
#make sure there is a cache folder in your scratch folder 
WANDB_CONFIG_DIR=$ROOT_DIR/cache/wandb/config
WANDB_CACHE_DIR=$ROOT_DIR/cache/wandb/cache
WANDB_DIR=$ROOT_DIR/cache/wandb/log
MASTER_NODE=$(head -n 1 $PBS_NODEFILE)
NNODES=$(cat $PBS_NODEFILE | wc -l)

mkdir -p $SCRATCH_DIR/output/vallex/exp/$EXP_NAME/logs

pbsdsh bash $CODE_DIR/enroot_start.sh \
    $HF_HOME \
    $ROOT_DIR \
    $CODE_DIR \
    $HF_HUB_OFFLINE \
    $WANDB_CONFIG_DIR \
    $WANDB_CACHE_DIR \
    $WANDB_DIR \
    $MASTER_NODE \
    $NNODES \
    $EXP_NAME \
    $EXP_PATH \
    $SCRATCH_DIR \
    $PRE_NAME \
    $STAGE \
    $START_EPOCH
