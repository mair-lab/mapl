dependencies = ['torch', 'transformers']

from typing import Any

import torch.nn as nn

from mapl import MAPL


def mapl(**kwargs: Any) -> nn.Module:
    checkpoint_url = 'https://github.com/mair-lab/mapl-private/raw/main/checkpoints/mapl-clip-vit-l14-gpt-j-coco.pt'
    return MAPL.from_pretrained(checkpoint_url, **kwargs)
