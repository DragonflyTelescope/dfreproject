import torch


def get_device():
    """
    Utility function to get the device

    Returns
    -------
    torch.device
        Available torch device (either gpu or cpu)
    """
    try:
        # Try to get CUDA device count to check if CUDA is properly initialized
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    except Exception as e:
        print(f"CUDA error detected: {e}")
        print("Falling back to CPU.")
        return torch.device("cpu")
