# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model fantastic_v0 \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --num_experts 16 \
#     --top_k 16 \
#     --d_state 64 \
#     --orthogonal_loss_coef 0.01 \
#     --model_name "fantastic-v0-130m-e16a16-s64-orth001" \
#     --max_val_batches 50 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model fantastic \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --num_experts 16 \
#     --top_k 16 \
#     --d_state 16 \
#     --basis_mode \
#     --orthogonal_loss_coef 0.01 \
#     --model_name "fantastic-130m-e16a16-orth001-sep-linspace" \
#     --separate_routing \
#     --dt_strategy linspace \
#     --max_val_batches 50 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model attentive \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --num_experts 16 \
#     --top_k 16 \
#     --d_state 16 \
#     --basis_mode \
#     --orthogonal_loss_coef 0.001 \
#     --orthogonal_loss_coef_dt 0.001 \
#     --model_name "attentive-130m-orth0001_0001" \
#     --max_val_batches 50 \
#     # --use_pretrained \
#     # --freeze_embeds \

#     # --separate_routing \

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model mamba \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --d_state 16 \
#     --model_name "mamba-130m-2" \
#     --max_val_batches 50 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model mamba \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --d_state 16 \
#     --model_name "mamba-130m-resume" \
#     --max_val_batches 50 \
#     --resume "/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/mamba-130m/step_0020000.pt" \
#     --train_offset 80000 \

    # --dt_rank "auto" \
    # --dt_num_experts 16 \
    # --dt_top_k 16 \
    # --lb_strategy "aux_free" \
    # --lb_coef 0.001 \
    # --dropout 0.1 \
    # --ff_mult 2 \
    # --model_name "s4d-s64"


    # --model_name "mamba-130m"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model s4d \
    --data_dir ./train_data/data/train \
    --val_dir  ./train_data/data/val  \
    --val_every 500 \
    --batch_size 8         \
    --grad_accum_steps 8   \
    --max_steps 100000     \
    --d_state 64 \
    --ff_mult 2 \
    --model_name "s4d-s64-resume-offset" \
    --max_val_batches 50 \
    --resume "/home/jovyan/shares/SR008.fs2/leopetrov/projects/fantastic/experiments/checkpoints/s4d-s64/step_0018000.pt" \
    --train_offset 144000 \