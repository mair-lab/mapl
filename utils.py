from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Any, Optional, Union

import torch


def accumulate_padding(input_embeds: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = 'right') -> Tuple[torch.Tensor, torch.Tensor]:
    assert padding_side in ['right', 'left']

    new_input_embeds = torch.empty_like(input_embeds)
    new_attention_masks = torch.empty_like(attention_mask)

    for i, (embed, mask) in enumerate(zip(input_embeds, attention_mask)):
        padding_indices = torch.where(mask == 0)[0]
        non_padding_indices = torch.where(mask == 1)[0]
        if padding_side == 'left':
            new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
        else:
            new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
        new_input_embeds[i] = embed.index_select(0, new_indices)
        new_attention_masks[i] = mask.index_select(0, new_indices)

    return new_input_embeds, new_attention_masks


class torch_dtype:
    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
    
    def __enter__(self) -> Any:
        self.dtype_orig = torch.get_default_dtype()
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Optional[bool]:
        if self.dtype is not None:
            torch.set_default_dtype(self.dtype_orig)


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def convert_lightning_checkpoint(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    checkpoint = torch.load(src_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    for k, v in hparams.items():
        print(f"{k}: {v}")
    state_dict = checkpoint['state_dict']
    torch.save(state_dict, dst_path)
