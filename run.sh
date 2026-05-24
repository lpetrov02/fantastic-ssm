CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model fantastic_v0 \
    --data_dir ./train_data/data/train \
    --val_dir  ./train_data/data/val  \
    --val_every 100000 \
    --batch_size 16         \
    --grad_accum_steps 4   \
    --max_steps 100000     \
    --num_experts 16 \
    --top_k 1 \
    --d_state 64 \
    --model_name "fantastic-v0-130m-e16a1-s64"

    # --lb_strategy "aux_free" \
    # --lb_coef 0.001 \
    # --dropout 0.1 \
    # --ff_mult 2 \
    # --model_name "s4d-s64"


    # --model_name "mamba-130m"
