import logging
import math
from typing import Tuple
import torch

logger = logging.getLogger(__name__)


def get_device():
    """
    Utility function to get the currently available PyTorch device.

    Returns
    -------
    torch.device
        Available torch device (either cuda or cpu).
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


def gradient2d(tensor):
    """
    Compute gradients (dy, dx) of a 2D tensor using centered differences.
    """
    dx = (
        torch.roll(tensor, shifts=-1, dims=-1) - torch.roll(tensor, shifts=1, dims=-1)
    ) / 2.0
    dy = (
        torch.roll(tensor, shifts=-1, dims=-2) - torch.roll(tensor, shifts=1, dims=-2)
    ) / 2.0

    # fix edges with forward/backward differences
    dx[..., 0] = tensor[..., 1] - tensor[..., 0]
    dx[..., -1] = tensor[..., -1] - tensor[..., -2]
    dy[..., 0, :] = tensor[..., 1, :] - tensor[..., 0, :]
    dy[..., -1, :] = tensor[..., -1, :] - tensor[..., -2, :]

    return dy, dx


def estimate_memory_per_pixel(reproject_instance, interpolation_mode: str) -> float:
    """
    Estimate memory usage per output pixel in bytes.

    Parameters
    ----------
    reproject_instance : Reproject
        An initialized Reproject instance.
    interpolation_mode : str
        Interpolation mode being used.

    Returns
    -------
    float
        Estimated bytes per pixel.
    """
    B = len(reproject_instance.batch_source_images)

    # Base memory for coordinates and intermediate tensors
    # Each pixel needs:
    # - x_source, y_source: 2 * 8 bytes (float64)
    # - x_normalized, y_normalized: 2 * 8 bytes
    # - combined_result: 2 channels * 8 bytes
    # - result: 8 bytes
    # - valid_pixels mask: 1 byte
    base_per_pixel = (2 + 2 + 2 + 1 + 1) * 8

    # Add memory for Jacobian if needed
    if reproject_instance.compute_jacobian and reproject_instance.conserve_flux:
        # 4 gradient tensors: dx_x, dy_x, dx_y, dy_y
        base_per_pixel += 4 * 8

    # Multiply by batch size
    total_per_pixel = base_per_pixel * B

    # Add overhead for intermediate operations (20% extra)
    total_per_pixel *= 1.2

    return total_per_pixel


def calculate_chunk_size(
    reproject_instance,
    output_shape: Tuple[int, int],
    max_memory_mb: float,
    safety_factor: float,
    interpolation_mode: str = "bilinear",
) -> Tuple[int, int]:
    """
    Calculate optimal chunk size based on memory constraints.

    Parameters
    ----------
    reproject_instance : Reproject
        An initialized Reproject instance.
    output_shape : Tuple[int, int]
        Shape of the output image (H, W).
    max_memory_mb : float
        Maximum memory to use in megabytes.
    safety_factor : float
        Safety factor (0-1) for memory calculation.
    interpolation_mode : str
        Interpolation mode to use.

    Returns
    -------
    Tuple[int, int]
        Chunk size (chunk_height, chunk_width).
    """
    H, W = output_shape

    # Available memory in bytes
    available_bytes = max_memory_mb * 1024 * 1024 * safety_factor

    # Memory per pixel
    bytes_per_pixel = estimate_memory_per_pixel(reproject_instance, interpolation_mode)

    # Maximum pixels per chunk
    max_pixels = int(available_bytes / bytes_per_pixel)

    # Ensure at least 1 row can be processed
    if max_pixels < W:
        logger.warning(
            f"Memory limit very tight! Estimated {bytes_per_pixel:.2f} bytes/pixel. "
            f"Consider increasing max_memory_mb (currently {max_memory_mb} MB)."
        )
        max_pixels = W

    # Calculate chunk dimensions
    # Try to make chunks roughly square for better cache performance
    chunk_height = min(H, int(math.sqrt(max_pixels * H / W)))
    chunk_height = max(1, chunk_height)  # At least 1 row

    # Calculate corresponding width
    chunk_width = min(W, max_pixels // chunk_height)
    chunk_width = max(1, chunk_width)

    return chunk_height, chunk_width


def process_chunk(
    reproject_instance,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    interpolation_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Process a single chunk of the reprojection.

    Parameters
    ----------
    reproject_instance : Reproject
        An initialized Reproject instance.
    y_start, y_end : int
        Y-axis range for the chunk.
    x_start, x_end : int
        X-axis range for the chunk.
    interpolation_mode : str
        Interpolation mode.

    Returns
    -------
    torch.Tensor
        Reprojected chunk.
    """
    # Create chunk-specific grid
    B = len(reproject_instance.batch_source_images)
    chunk_h = y_end - y_start
    chunk_w = x_end - x_start

    y_chunk = (
        torch.arange(
            y_start, y_end, dtype=torch.float64, device=reproject_instance.device
        )
        .view(1, -1, 1)
        .expand(B, chunk_h, chunk_w)
    )

    x_chunk = (
        torch.arange(
            x_start, x_end, dtype=torch.float64, device=reproject_instance.device
        )
        .view(1, 1, -1)
        .expand(B, chunk_h, chunk_w)
    )

    # Temporarily override the target grid
    original_grid = reproject_instance.target_grid
    reproject_instance.target_grid = (y_chunk, x_chunk)

    try:
        # Process this chunk
        chunk_result = reproject_instance.interpolate_source_image(
            interpolation_mode=interpolation_mode
        )
    finally:
        # Restore original grid
        reproject_instance.target_grid = original_grid

    # Clear cache
    if reproject_instance.device.type == "cuda":
        torch.cuda.empty_cache()

    return chunk_result


def reproject_chunked(
    reproject_instance,
    max_memory_mb: float,
    safety_factor: float,
    interpolation_mode: str = "bilinear",
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Perform chunked reprojection.

    Parameters
    ----------
    reproject_instance : Reproject
        An initialized Reproject instance.
    max_memory_mb : float
        Maximum memory to use in megabytes.
    safety_factor : float
        Safety factor for memory calculation.
    interpolation_mode : str
        Interpolation mode to use.
    show_progress : bool
        Whether to log progress information.

    Returns
    -------
    torch.Tensor
        Full reprojected image.
    """
    y_grid, x_grid = reproject_instance.target_grid
    B, H, W = y_grid.shape

    # Calculate chunk size
    chunk_h, chunk_w = calculate_chunk_size(
        reproject_instance, (H, W), max_memory_mb, safety_factor, interpolation_mode
    )

    # Calculate number of chunks
    n_chunks_y = math.ceil(H / chunk_h)
    n_chunks_x = math.ceil(W / chunk_w)
    total_chunks = n_chunks_y * n_chunks_x

    # Initialize output
    result = torch.full(
        (B, H, W), torch.nan, dtype=torch.float64, device=reproject_instance.device
    )

    # Process chunks
    chunk_idx = 0
    for i in range(n_chunks_y):
        y_start = i * chunk_h
        y_end = min((i + 1) * chunk_h, H)

        for j in range(n_chunks_x):
            x_start = j * chunk_w
            x_end = min((j + 1) * chunk_w, W)

            if show_progress:
                chunk_idx += 1

            # Process chunk
            chunk_result = process_chunk(
                reproject_instance, y_start, y_end, x_start, x_end, interpolation_mode
            )

            # Insert into result
            result[:, y_start:y_end, x_start:x_end] = chunk_result

    return result
