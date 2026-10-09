import torch
import numpy as np
import random
from typing import Optional

def set_seed(seed: Optional[int]):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # If CUDA is available, set seed for it too
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # For MPS, manual_seed sets it for all devices, but we ensure determinism as much as possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
    else:
        print("No random seed provided. Execution may not be perfectly reproducible.")
