python train.py --train-stage 1 \
      --save-every-n 10000 --valid-interval 20000 \
      --model-name vallex \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --train_dir /scratch/users/astar/ares/ma_yi/output/vallex/ \
      --valid_dir /scratch/users/astar/ares/ma_yi/output/vallex/ \
      --lang zh \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir /scratch/users/astar/ares/ma_yi/output/vallex/exp
