# fantastic-ssm

## Training

```bash
# Single node, 4 GPUs — FantasticSSM
torchrun --standalone --nproc_per_node=4 train/train.py \
    --model fantastic \
    --data_dir ./data/train \
    --val_dir  ./data/val  \
    --batch_size 8         \
    --grad_accum_steps 4   \
    --max_steps 100000     \
    --checkpoint_dir ./checkpoints

# Resume from a checkpoint
torchrun --standalone --nproc_per_node=4 train/train.py \
    --resume ./checkpoints/step_0010000.pt [other args…]
```

TensorBoard logs are written to `./checkpoints/tensorboard/`.  
Checkpoints go to `./checkpoints/step_NNNNNNN.pt` (keeps the last 3).

## Setup

Install the project in editable mode from the repo root so that `models` and its subpackages are importable from anywhere:

```bash
python -m pip install -e .
```

After that you can import from `models` in any file:

```python
from models.mamba.mamba import Mamba
from models.s4.s4d import S4D
from models.attention.mha import MultiheadAttention
from models.modules.nn import GatedMLP, DropoutNd
from models.model import Model
```