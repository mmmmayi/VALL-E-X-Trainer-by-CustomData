HF_HOME=$1
ROOT_DIR=$2
CODE_DIR=$3
HF_HUB_OFFLINE=$4
WANDB_CONFIG_DIR=$5
WANDB_CACHE_DIR=$6
WANDB_DIR=$7
MASTER_NODE=$8
NNODES=${9}
EXP_NAME=${10}
EXP_PATH=${11}
SCRATCH_DIR=${12}

torchrun_command="
	enroot start \
		-r -w \
		-m $ROOT_DIR:/home/users/astar/ares/ma_yi \
		-m $SCRATCH_DIR:/scratch/users/astar/ares/ma_yi \
		-m /raid/local/containers/enroot-data/$PBS_JOBID/tts/tmp/:/dev/shm \
		-e HF_HOME=$HF_HOME \
		-e WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR \
		-e WANDB_CACHE_DIR=$WANDB_CACHE_DIR \
		-e WANDB_DIR=$WANDB_DIR \
		-e HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
		-e HF_ENDPOINT=https://huggingface.co \
		-e TMPDIR=/mnt/scratch/tmp \
		-e TEMP=/mnt/scratch/tmp \
		-e TORCH_CPP_LOG_LEVEL=ERROR \
		tts \
		bash -c \"
		cd $CODE_DIR && \
		torchrun \
			--nnodes=$NNODES \
			--nproc_per_node=8 \
			--rdzv_id=job_$PBS_JOBID \
			--rdzv_backend=c10d \
			--rdzv_endpoint=$MASTER_NODE:35000 \
			--log_dir=/mnt/scratch/log \
				train.py \
					--train-stage 1 \
					--save-every-n 100 --valid-interval 20000 \
					--model-name vallex \
					--base-lr 0.05 --warmup-steps 200 --average-period 0 \
					--train_dir /scratch/users/astar/ares/ma_yi/output/vallex/ \
					--valid_dir /scratch/users/astar/ares/ma_yi/output/vallex/ \
					--lang zh \
					--num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
					--exp-dir $EXP_PATH/$EXP_NAME\"
"



if [ "$HOSTNAME" = "$(echo $MASTER_NODE | cut -d'.' -f1)" ]; then
  eval ${torchrun_command}
else
  eval ${torchrun_command} > ${EXP_PATH}/${EXP_NAME}/${HOSTNAME}.log 2>&1
fi