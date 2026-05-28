# CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
#     --model_name "fantastic-130m-e16a16-orth001-ns-pretrained" \
#     --use_pretrained \
#     --freeze_embeds \
#     --max_val_batches 50 \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model attentive \
    --data_dir ./train_data/data/train \
    --val_dir  ./train_data/data/val  \
    --val_every 500 \
    --batch_size 16         \
    --grad_accum_steps 4   \
    --max_steps 100000     \
    --num_experts 16 \
    --top_k 16 \
    --d_state 16 \
    --basis_mode \
    --orthogonal_loss_coef 0.01 \
    --orthogonal_loss_coef_dt 0.0001 \
    --model_name "attentive-130m-pretrained" \
    --use_pretrained \
    --freeze_embeds \
    --max_val_batches 50 \

    # --separate_routing \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --standalone --nproc_per_node=4 train/train.py \
#     --model mamba \
#     --data_dir ./train_data/data/train \
#     --val_dir  ./train_data/data/val  \
#     --val_every 500 \
#     --batch_size 16         \
#     --grad_accum_steps 4   \
#     --max_steps 100000     \
#     --d_state 16 \
#     --model_name "mamba-130m-pretrained" \
#     --use_pretrained \
#     --freeze_embeds \
#     --max_val_batches 50 \

    # --dt_rank "auto" \
    # --dt_num_experts 16 \
    # --dt_top_k 16 \
    # --lb_strategy "aux_free" \
    # --lb_coef 0.001 \
    # --dropout 0.1 \
    # --ff_mult 2 \
    # --model_name "s4d-s64"


    # --model_name "mamba-130m"
