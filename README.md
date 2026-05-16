# fantastic-ssm

## Setup

Install the project in editable mode from the repo root so that `models` and its subpackages are importable from anywhere:

```bash
pip install -e .
```

After that you can import from `models` in any file:

```python
from models.mamba.mamba import Mamba
from models.s4.s4d import S4D
from models.attention.mha import MultiheadAttention
from models.modules.nn import GatedMLP, DropoutNd
from models.model import Model
```