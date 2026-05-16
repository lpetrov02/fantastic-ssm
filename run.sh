CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model fantastic \
    --data_dir ./train_data/data/train \
    --val_dir  ./train_data/data/val  \
    --batch_size 16         \
    --grad_accum_steps 4   \
    --max_steps 100000     \
    --model_name "fantastic-130m"