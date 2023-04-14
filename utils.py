from urllib.parse import urlparse
from typing import Any, Optional

import torch


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
