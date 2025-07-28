import logging

import torch

logger = logging.getLogger(__name__)


def get_device():
    """
    Utility function to get the currently available PyTorch device

    Returns
    -------
    torch.device
        Available torch device (either cuda or cpu)
    """
    try:
        # Try to get CUDA device count to check if CUDA is properly initialized
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    except Exception as e:
        logger.warning(f"CUDA error: {e}. Falling back to CPU.")
        return torch.device("cpu")
