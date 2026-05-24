CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model fantastic \
    --data_dir ./train_data/data/train \
    --val_dir  ./train_data/data/val  \
    --val_every 100000 \
    --batch_size 16         \
    --grad_accum_steps 4   \
    --max_steps 100000     \
    --num_experts 1 \
    --top_k 1 \
    --model_name "fantastic-130m-e1a1"

    # --dt_strategy logspace \
    # --lb_strategy "aux_free" \
    # --lb_coef 0.001 \
    # --dropout 0.1 \
    # --ff_mult 2 \
    # --d_state 64 \
    # --model_name "s4d-s64"


    # --model_name "mamba-130m"
