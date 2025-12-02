import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.io.fits import Header, PrimaryHDU
from astropy.wcs import WCS

from .lanczos import lanczos_grid_sample
from .sip import apply_inverse_sip_distortion, apply_sip_distortion, get_sip_coeffs
from .tensorhdu import TensorHDU
from .utils import get_device, gradient2d, reproject_chunked

logger = logging.getLogger(__name__)

EPSILON = 1e-10
VALID_ORDERS = ["bicubic", "bilinear", "nearest", "nearest-neighbors", "lanczos"]


def validate_interpolation_order(order: str) -> str:
    """
    Function to validate the requested interpolation order.

    The order must be one of the following: "bicubic", "bilinear",
    "nearest-neighbors". "nearest" is an alias for "nearest-neighbors".

    Parameters
    ----------
    order : str
        Interpolation order to validate.

    Returns
    -------
    str
        Validated interpolation order.

    Raises
    ------
    ValueError
        When the provided order is not one of the valid interpolation orders.
    """
    if order not in VALID_ORDERS:
        raise ValueError(f"order must be one of: {', '.join(VALID_ORDERS)}")
    elif order == "nearest-neighbors":
        return "nearest"
    else:
        return order


# Helper functions for trigonometric calculations
def atan2d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of WCSLib's atan2d function.

    Parameters
    ----------
    y : torch.Tensor
        y coordinate(s).
    x : torch.Tensor
        x coordinate(s).

    Returns
    -------
    torch.Tensor
        atan2d(y, x) in degrees.
    """
    return torch.rad2deg(torch.atan2(y, x))


def sincosd(angle_deg: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of WCSLib's sincosd function.

    Parameters
    ----------
    angle_deg : torch.Tensor
        angle in degrees.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor)
        sin(angle) in degrees, cos(angle) in degrees.
    """
    angle_rad = torch.deg2rad(angle_deg)
    return torch.sin(angle_rad), torch.cos(angle_rad)


def interpolate_image(
    source_image: torch.Tensor, grid: torch.Tensor, interpolation_mode: str
) -> torch.Tensor:
    """
    Image interpolation using grid_sample with LANCZOS support.

    Parameters
    ----------
    source_image : torch.Tensor
        Source image to interpolate.
    grid : torch.Tensor
        Grid on which to interpolate.
    interpolation_mode: str
        Interpolation mode to use. Supports PyTorch's built-in modes
        ('bilinear', 'bicubic', 'nearest') plus 'lanczos' for LANCZOS-3.

    Returns
    -------
    torch.Tensor
        Interpolated image.
    """
    if interpolation_mode == "lanczos":
        return lanczos_grid_sample(source_image, grid, padding_mode="zeros")
    else:
        return torch.nn.functional.grid_sample(
            source_image,
            grid,
            mode=interpolation_mode,
            align_corners=True,
            padding_mode="zeros",
        )


class Reproject:
    def __init__(
        self,
        source_hdus: List[PrimaryHDU],
        target_wcs: WCS,
        shape_out: Tuple[int, int],
        device: str = None,
        num_threads: int = None,
        requires_grad: bool = False,
        conserve_flux: bool = True,
        compute_jacobian: bool = True,
    ):
        """
        Initialize a dfreproject operation between source and target image frames.

        This constructor sets up the necessary components for reprojecting an astronomical
        image from one World Coordinate System (WCS) to another. It stores the source
        and target WCS information, images, and creates a coordinate grid for the target
        image that will be used in the dfreproject process.

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the
            source image.

        target_wcs : WCS
            WCS for the target in an astropy.wcs compatible format.

        shape_out: Tuple[int, int]
            Shape of the output image.

        device: str
            Device to use for computations. Defaults to GPU if available
            otherwise uses CPU.

        num_threads: int
            Number of threads to use on CPU.

        conserve_flux: bool
            If True, enables flux conservation through footprint calculation.

        compute_jacobian: bool, optional
            If True, enables non-linear flux conservation through Jacobian calculation.
            Note that this increases RAM usage.

        Notes
        -----
        This constructor creates a coordinate grid spanning the entire target image,
        which will be used for the pixel-to-world and world-to-pixel transformations
        during dfreproject. The grid is created with 'ij' indexing, where the first
        dimension corresponds to y (rows) and the second to x (columns).

        The coordinate grid is stored as a tuple of tensors (batch, y_grid, x_grid), where
        each element has the same shape as the target image.

        Examples
        --------
        >>> # Initialize the dfreproject object
        >>> reproject = Reproject(source_hdus, target_wcs)
        """

        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        if num_threads:
            torch.set_num_threads(num_threads)

        self.requires_grad = requires_grad

        self.batch_source_images = self._prepare_source_images(source_hdus)

        # Initialize the WCS objects
        self.batch_source_wcs_params = self._prepare_batch_wcs_params(source_hdus)
        self.target_wcs_params = self._extract_wcs_params(target_wcs)
        self.target_wcs = target_wcs

        # Define target grid
        self.target_grid = self._create_batch_target_grid(shape_out)

        # Define flux conservation booleans
        self.conserve_flux = conserve_flux
        self.compute_jacobian = compute_jacobian

    def _prepare_source_images(self, source_hdus: List[PrimaryHDU]) -> torch.Tensor:
        """
        Prepare batch of source images as a single tensor.

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the source image.

        Returns
        -------
        source_image : torch.Tensor
            Stack of source image tensors.
        """
        try:
            source_images = []
            for hdu in source_hdus:
                if self.requires_grad and isinstance(hdu, TensorHDU):
                    img = hdu.tensor.to(self.device)
                else:
                    img = torch.tensor(
                        hdu.data, dtype=torch.float64, device=self.device
                    )
                source_images.append(img)

        except ValueError:  # In case there is a byte order error
            source_images = [
                torch.tensor(
                    np.asarray(hdu.data, dtype=np.float64).copy(),
                    dtype=torch.float64,
                    device=self.device,
                )
                for hdu in source_hdus
            ]
        return torch.stack(source_images)

    def _extract_wcs_params(self, wcs: WCS) -> dict:
        """
        Extract key WCS parameters into a dictionary for efficient tensor operations.

        Returns a dictionary with pre-computed tensor parameters.

        Parameters
        ----------
        wcs : WCS
            WCS information.

        Returns
        -------
        wcs_params : dict
            WCS parameters.
        """
        return {
            "crpix": torch.tensor(
                wcs.wcs.crpix, dtype=torch.float64, device=self.device
            ),
            "crval": torch.tensor(
                wcs.wcs.crval, dtype=torch.float64, device=self.device
            ),
            "pc_matrix": torch.tensor(
                wcs.wcs.get_pc(), dtype=torch.float64, device=self.device
            ),
            "cdelt": torch.tensor(
                wcs.wcs.cdelt, dtype=torch.float64, device=self.device
            ),
            "sip_coeffs": get_sip_coeffs(wcs),
        }

    def _prepare_batch_wcs_params(
        self, source_hdus: Union[List[PrimaryHDU], List[TensorHDU]]
    ) -> List[dict]:
        """
        Prepare batch of WCS parameters.

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the
            source image.

        Returns
        -------
        List[dict]
            List of dictionaries containing the WCS parameters extracted from each HDU.
        """
        return [self._extract_wcs_params(WCS(hdu.header)) for hdu in source_hdus]

    def _create_batch_target_grid(
        self, shape_out: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a batched target grid matching the number of source images.

        Parameters
        ----------
        shape_out : Tuple[int, int]
            Shape of the output image.
        """
        B = len(self.batch_source_images)
        H, W = shape_out

        # Create base grids once
        y_base = torch.arange(H, dtype=torch.float64, device=self.device)
        x_base = torch.arange(W, dtype=torch.float64, device=self.device)

        # Use broadcasting instead of repeat to save memory during creation
        # expand() creates a view without allocating new memory
        y_grid = y_base.view(1, -1, 1).expand(B, H, W)
        x_grid = x_base.view(1, 1, -1).expand(B, H, W)

        # Only make contiguous copies if needed for operations
        # Most operations work fine with non-contiguous tensors
        return y_grid, x_grid

    def calculate_skyCoords(
        self, x_grid=None, y_grid=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate sky coordinates.

        There are four primary steps:
        1. Apply shift
        2. Apply SIP distortion
        3. Apply CD matrix
        4. Apply transformation to celestial coordinates uing the Gnomonic Projection

        These steps use the target wcs parameters.

        Parameters
        ----------
        x_grid : torch.Tensor, optional
            Batch of x-coordinates. If None, uses target grid x-coordinates.
        y_grid : torch.Tensor, optional
            Batch of y-coordinates. If None, uses target grid y-coordinates.

        Returns
        -------
        tuple
            Batched RA and Dec coordinates.
        """

        if x_grid is None or y_grid is None:
            y_grid, x_grid = self.target_grid
        else:
            if not isinstance(x_grid, torch.Tensor):
                x_grid = torch.tensor(x_grid, dtype=torch.float64, device=self.device)
            if not isinstance(y_grid, torch.Tensor):
                y_grid = torch.tensor(y_grid, dtype=torch.float64, device=self.device)
            if x_grid.dim() == 2:
                x_grid = x_grid.unsqueeze(0)
            if y_grid.dim() == 2:
                y_grid = y_grid.unsqueeze(0)

        B, H, W = y_grid.shape

        # Unpack target WCS parameters
        crpix = self.target_wcs_params["crpix"]
        crval = self.target_wcs_params["crval"]
        pc_matrix = self.target_wcs_params["pc_matrix"]
        cdelt = self.target_wcs_params["cdelt"]
        sip_coeffs = self.target_wcs_params["sip_coeffs"]

        # Compute pixel offsets (in-place when possible)
        u = x_grid - (crpix[0] - 1)
        v = y_grid - (crpix[1] - 1)

        # Apply SIP distortion if present
        if sip_coeffs is not None:
            u, v = apply_sip_distortion(u, v, sip_coeffs, self.device)

        # Apply PC Matrix and CDELT
        CD_matrix = pc_matrix * cdelt

        # Optimize matrix multiplication
        pixel_offsets = torch.stack([u.reshape(B, -1), v.reshape(B, -1)], dim=-1)
        del u, v  # Free immediately

        transformed = torch.bmm(
            pixel_offsets, CD_matrix.T.unsqueeze(0).expand(B, -1, -1)
        )
        del pixel_offsets, CD_matrix

        x_scaled = transformed[:, :, 0].reshape(B, H, W)
        y_scaled = transformed[:, :, 1].reshape(B, H, W)
        del transformed

        # Compute radial distance (in-place operations)
        r = torch.sqrt(x_scaled.pow(2).add_(y_scaled.pow(2)))  # More memory efficient
        r0 = torch.tensor(180.0 / torch.pi)

        # Compute phi efficiently
        phi = torch.zeros_like(r)
        non_zero_r = r > 0  # Avoid creating unnecessary boolean tensor
        phi[non_zero_r] = torch.rad2deg(
            torch.atan2(-x_scaled[non_zero_r], y_scaled[non_zero_r])
        )
        del x_scaled, y_scaled, non_zero_r

        phi_rad = torch.deg2rad(phi)
        del phi

        theta_rad = torch.atan2(
            r0, r
        )  # Direct computation without intermediate conversions
        del r

        # Pre-compute trig values
        ra0_rad = crval[0] * (torch.pi / 180.0)
        dec0_rad = crval[1] * (torch.pi / 180.0)

        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)
        del theta_rad, phi_rad

        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)

        # Compute dec
        sin_dec = sin_theta * sin_dec0 + cos_theta * cos_dec0 * cos_phi
        dec_rad = torch.arcsin(sin_dec)
        del sin_dec

        # Compute ra (reuse tensors where possible)
        ra_rad = ra0_rad + torch.atan2(
            -cos_theta * sin_phi, sin_theta * cos_dec0 - cos_theta * sin_dec0 * cos_phi
        )
        del sin_theta, cos_theta, sin_phi, cos_phi, sin_dec0, cos_dec0

        # Convert to degrees
        ra = torch.rad2deg(ra_rad) % 360.0
        dec = torch.rad2deg(dec_rad)
        del ra_rad, dec_rad

        return ra, dec

    def calculate_sourceCoords(self):
        """
        Calculate source image pixel coordinates corresponding to each target image pixel.

        This function repeats the same steps in self.calculate_skyCoords()
        except in the opposite order and with the source coordinate wcs.

        Returns
        -------
        torch.Tensor
            Batch of source image pixel coordinates.
        """

        B = len(self.batch_source_images)
        y_grid, x_grid = self.target_grid
        _, H, W = y_grid.shape

        # Pre-allocate output tensors
        batch_x_pixel = torch.zeros((B, H, W), dtype=torch.float64, device=self.device)
        batch_y_pixel = torch.zeros((B, H, W), dtype=torch.float64, device=self.device)

        # Process each source image's coordinates
        for b in range(B):
            # Calculate sky coords for just this batch element
            # Extract single batch element from grid
            x_grid_b = x_grid[b : b + 1]  # Keep batch dimension
            y_grid_b = y_grid[b : b + 1]

            ra, dec = self.calculate_skyCoords(x_grid_b, y_grid_b)
            ra = ra.squeeze(0)  # Remove batch dimension
            dec = dec.squeeze(0)

            # Free grid slices immediately
            del x_grid_b, y_grid_b

            # Get WCS parameters for this specific source image
            source_wcs_params = self.batch_source_wcs_params[b]
            crpix = source_wcs_params["crpix"]
            crval = source_wcs_params["crval"]
            pc_matrix = source_wcs_params["pc_matrix"]
            cdelt = source_wcs_params["cdelt"]
            sip_coeffs = source_wcs_params["sip_coeffs"]

            # Conversion calculations
            ra_rad = torch.deg2rad(ra)
            dec_rad = torch.deg2rad(dec)
            ra0_rad = crval[0] * torch.pi / 180.0
            dec0_rad = crval[1] * torch.pi / 180.0

            # Convert from world to native spherical coordinates
            y_phi = -torch.cos(dec_rad) * torch.sin(ra_rad - ra0_rad)
            x_phi = torch.sin(dec_rad) * torch.cos(dec0_rad) - torch.cos(
                dec_rad
            ) * torch.sin(dec0_rad) * torch.cos(ra_rad - ra0_rad)
            phi = torch.rad2deg(torch.atan2(y_phi, x_phi))
            del x_phi, y_phi

            theta = torch.rad2deg(
                torch.arcsin(
                    torch.sin(dec_rad) * torch.sin(dec0_rad)
                    + torch.cos(dec_rad)
                    * torch.cos(dec0_rad)
                    * torch.cos(ra_rad - ra0_rad)
                )
            )
            del ra_rad, dec_rad, ra, dec

            # Apply TAN projection
            sin_phi, cos_phi = (
                torch.sin(torch.deg2rad(phi)),
                torch.cos(torch.deg2rad(phi)),
            )
            del phi
            sin_theta, cos_theta = (
                torch.sin(torch.deg2rad(theta)),
                torch.cos(torch.deg2rad(theta)),
            )
            del theta

            # Check for singularity
            eps = 1e-10
            if torch.any(torch.abs(sin_theta) < eps):
                raise ValueError("Singularity in tans2x: theta close to 0 degrees")

            r0 = torch.tensor(180.0 / torch.pi, device=self.device)
            r = r0 * cos_theta / sin_theta
            del cos_theta, sin_theta, r0

            x_scaled = -r * sin_phi
            y_scaled = r * cos_phi
            del sin_phi, cos_phi, r

            # Apply inverse CD matrix
            CD_matrix = pc_matrix * cdelt
            CD_inv = torch.linalg.inv(CD_matrix)
            del CD_matrix

            # Batch matrix multiplication
            x_scaled_flat = x_scaled.reshape(-1)
            y_scaled_flat = y_scaled.reshape(-1)
            del x_scaled, y_scaled

            standard_coords = torch.stack([x_scaled_flat, y_scaled_flat], dim=1)
            del x_scaled_flat, y_scaled_flat

            pixel_offsets = torch.matmul(standard_coords, CD_inv.T)
            u = pixel_offsets[:, 0].reshape(H, W)
            v = pixel_offsets[:, 1].reshape(H, W)
            del CD_inv, pixel_offsets, standard_coords

            if sip_coeffs is not None:
                u, v = apply_inverse_sip_distortion(u, v, sip_coeffs, self.device)

            # Add reference pixel
            batch_x_pixel[b] = u + (crpix[0] - 1)
            batch_y_pixel[b] = v + (crpix[1] - 1)
            del u, v

            # Clear references to WCS parameters to aid garbage collection
            del crpix, crval, pc_matrix, cdelt, sip_coeffs, source_wcs_params

        return batch_x_pixel, batch_y_pixel

    def interpolate_source_image(self, interpolation_mode="bilinear") -> torch.Tensor:
        """
        Interpolate the source image at the calculated source coordinates with flux conservation.

        This method performs the actual pixel resampling needed for dfreproject
        while preserving the total flux (photometric accuracy) by using a footprint correction and the Jacobian of the transformation.

        The method uses a combined tensor approach for computational efficiency,
        performing both image resampling and footprint tracking in a single operation.
        Total flux is conserved locally (via footprint correction and the Jacobian SIP calculation).

        Parameters
        ----------
        interpolation_mode : str, default 'bilinear'
            The interpolation mode to use when sampling the source image.
            Options include:
            - 'nearest' : Nearest neighbor interpolation (no interpolation)
            - 'bilinear' : Bilinear interpolation (default)
            - 'bicubic' : Bicubic interpolation
            - 'lanczos' : Lanczos interpolation

            These correspond to the modes available in torch.nn.functional.grid_sample.

        Returns
        -------
        torch.Tensor
            The reprojected image with the same shape as the target image.
            Pixel values are interpolated from the source image according to
            the WCS transformation with flux conservation preserved.

        Notes
        -----
        This implementation uses a two-step flux conservation approach:

        1. Local flux density conservation: The image and a "ones" tensor are interpolated
           together, and the interpolated image is divided by the interpolated ones
           tensor (footprint) to correct for any flux density spreading during interpolation.
           This is important when for pixels at the edge of the input image when mapped to the output
           image in case the input image only partially fills the output pixel.
        2. Jacobian correction for full flux conservation: Multiply the footprint-corrected flux
           by the determinant of the Jacobian to handle changes in area during the reprojection

        The Jacobian correction can be circumvented if you set
        compute_jacobian=False. However, the default behavior is to include this.

        Areas in the target image that map outside the source image boundaries
        will be filled with NaNs.
        """
        # Get source coordinates
        x_source, y_source = self.calculate_sourceCoords()
        B, H, W = self.batch_source_images.shape

        # Normalize coordinates
        x_normalized = 2.0 * (x_source / (W - 1)) - 1.0
        y_normalized = 2.0 * (y_source / (H - 1)) - 1.0

        # Prepare images and grid for grid_sample
        source_images = self.batch_source_images.unsqueeze(1)  # [B, 1, H, W]
        ones = torch.ones_like(source_images)

        # Combine images with ones for footprint calculation
        combined_result = interpolate_image(
            torch.cat([source_images, ones], dim=1),
            torch.stack([x_normalized, y_normalized], dim=-1),
            interpolation_mode,
        )
        del source_images, ones, x_normalized, y_normalized

        # Create output array initialized with NaN
        result = torch.full_like(combined_result[:, 0].squeeze(), torch.nan)
        valid_pixels = combined_result[:, 1].squeeze() > EPSILON
        # Apply footprint correction only where footprint is significant
        if torch.any(valid_pixels):
            if self.conserve_flux:
                # Normalize by the footprint where valid
                result[valid_pixels] = (
                    combined_result[:, 0].squeeze()[valid_pixels]
                    / combined_result[:, 1].squeeze()[valid_pixels]
                )

                # Free combined_result immediately
                del combined_result

                if self.compute_jacobian:
                    # Get gradients
                    dy_x, dx_x = gradient2d(x_source)  # ∂x_in/∂y_out, ∂x_in/∂x_out
                    dy_y, dx_y = gradient2d(y_source)  # ∂y_in/∂y_out, ∂y_in/∂x_out
                    jacobian_det = dx_x * dy_y
                    del dy_y
                    jacobian_det -= dy_x * dx_y
                    del dx_x, dy_x, dx_y
                    result[valid_pixels] *= jacobian_det.squeeze(0)[valid_pixels]
                    del jacobian_det
            else:
                result[valid_pixels] = combined_result[:, 0].squeeze()[valid_pixels]
                del combined_result
        else:
            result = combined_result[:, 0].squeeze() / combined_result[:, 1].squeeze()
            del combined_result
            logger.warning(
                "No valid pixels found in footprint! Using raw interpolated values"
            )

        del valid_pixels, x_source, y_source

        return result


def calculate_reprojection(
    source_hdus: Union[
        PrimaryHDU,
        TensorHDU,
        Tuple[np.ndarray, Union[WCS, Header]],
        Tuple[torch.Tensor, Union[WCS, Header]],
        List[Union[PrimaryHDU, Tuple[np.ndarray, Union[WCS, Header]]]],
    ],
    target_wcs: Union[WCS, Header],
    shape_out: Optional[Tuple[int, int]] = None,
    order: str = "nearest",
    device: str = None,
    num_threads: int = None,
    requires_grad: bool = False,
    conserve_flux: bool = True,
    compute_jacobian: bool = True,
    max_memory_mb: Optional[float] = None,
    chunk_safety_factor: float = 0.8,
    show_chunk_progress: bool = True,
    show_log: bool = False,
):
    """
    Reproject an astronomical image from a source WCS to a target WCS.

    This high-level function provides a convenient interface for image reprojection,
    handling all the necessary steps: WCS extraction, tensor creation, and interpolation.
    It converts FITS HDU objects to the internal representation, performs the reprojection,
    and returns the resulting image as a NumPy array or PyTorch tensor.

    Parameters
    ----------

    source_hdus : PrimaryHDU, TensorHDU, tuple, or list
        The source image(s) to be reprojected. Can be:
            - A PrimaryHDU
            - A TensorHDU
            - A tuple of (np.ndarray or torch.Tensor, WCS or Header)
            - A list of any of the above


    target_wcs : Union[WCS, Header]
        WCS information for the target. If a Header is passed it will be converted to WCS.

    shape_out: Optional[Tuple[int, int]]
        Shape of the resampled array. If not provided, the output shape will match the input.

    order : str, default 'nearest'
        The interpolation method to use when resampling the source image.
        Options:
        - 'nearest' : Nearest neighbor interpolation (fastest, default)
        - 'bilinear' : Bilinear interpolation (good balance of speed/quality)
        - 'bicubic' : Bicubic interpolation (high quality, slow)
        - 'lanczos' : Lanczos 3-lobe interpolation (highest quality, slowest)

    device: str, optional
        Device to use for computations. Defaults to GPU if available, otherwise uses CPU.

    num_threads: int, optional
        Number of threads to use on CPU.

    requires_grad: bool, optional
        If True, enables autograd for PyTorch tensors.

    conserve_flux: bool, optional
        If True, enables flux conservation through footprint calculations.
        By default, this is set to True.

    compute_jacobian: bool, optional
        If True, enables non-linear flux conservation through Jacobian calculation. Note
        that this slightly increases RAM usage.
        By default, this is set to True.
        If there is no SIP distortion, users can set this to False.

    max_memory_mb: Optional[float], optional
        Maximum memory to use in megabytes for chunked processing. If None (default),
        processes the entire image at once without chunking. Set this to enable
        memory-limited chunked processing (e.g., 1000 for 1GB limit).

    chunk_safety_factor: float, optional
        Safety factor (0-1) for chunked processing. Default 0.8 means use 80% of
        max_memory_mb for actual data, leaving 20% margin. Only used if max_memory_mb is set.

    show_chunk_progress: bool, optional
        Whether to log progress when using chunked processing. Default True.
        Only used if max_memory_mb is set.

    show_log: bool, optional
        Whether to log progress. Default True.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The reprojected image as a numpy ndarray (default) or PyTorch tensor if
        requires_grad=True.

    Notes
    -----
    This function automatically:
    - Detects and uses GPU acceleration if available
    - Handles byte order conversion for tensor creation
    - Converts data to float64 for processing
    - Converts Header to WCS if needed
    - Processes in memory-constrained chunks if max_memory_mb is specified

    **Chunked Processing:**
    When max_memory_mb is set, the reprojection is computed in blocks to stay within
    the specified memory limit. This is useful for:
    - Very large output images
    - Limited GPU memory
    - Batch processing multiple images


    To save the result as a FITS file, convert the tensor back to a NumPy array
    and create a new FITS HDU with the target WCS header.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from astropy.wcs import WCS
    >>> from dfreproject.reproject import calculate_reprojection
    >>>
    >>> # Open source and target images
    >>> source_hdu = fits.open('source_image.fits')[0]
    >>> target_hdu = fits.open('target_grid.fits')[0]
    >>> target_wcs = WCS(target_hdu.header)
    >>>
    >>> # Perform reprojection with bilinear interpolation
    >>> reprojected = calculate_reprojection(
    ...     source_hdus=source_hdu,
    ...     target_wcs=target_wcs,
    ...     shape_out=target_hdu.data.shape,
    ...     order='bilinear'
    ... )
    >>>
    >>> # Perform chunked reprojection with 2GB memory limit
    >>> reprojected = calculate_reprojection(
    ...     source_hdus=source_hdu,
    ...     target_wcs=target_wcs,
    ...     shape_out=(8000, 8000),
    ...     order='bilinear',
    ...     max_memory_mb=2000
    ... )
    >>> # Save as FITS
    >>> output_hdu = fits.PrimaryHDU(data=reprojected, header=target_hdu.header)
    >>> output_hdu.writeto('reprojected_image.fits', overwrite=True)
    """

    def normalize_to_hdu(item):
        if isinstance(item, PrimaryHDU):
            if requires_grad and not isinstance(item, TensorHDU):
                return TensorHDU(data=item.data, header=item.header)
            else:
                return item
        elif isinstance(item, tuple) and len(item) == 2:
            data, wcs_or_header = item
            if isinstance(wcs_or_header, Header):
                header = wcs_or_header
            elif isinstance(wcs_or_header, WCS):
                header = wcs_or_header.to_header(relax=True)
            else:
                raise TypeError("Expected WCS or Header in tuple.")
            if requires_grad:
                return TensorHDU(data=data, header=header)
            else:
                return PrimaryHDU(data=data, header=header)
        else:
            raise TypeError(
                "Each item must be a PrimaryHDU, TensorHDU, or a (data, wcs/header) tuple."
            )

    # Normalize source_input to a list of HDUs
    if isinstance(source_hdus, list):
        source_hdus = [normalize_to_hdu(item) for item in source_hdus]
    else:
        source_hdus = [normalize_to_hdu(source_hdus)]

    # Convert Header to WCS if needed
    if isinstance(target_wcs, Header):
        target_wcs = WCS(target_wcs)
    if not shape_out:
        shape_out = source_hdus[0].data.shape

    reprojection = Reproject(
        source_hdus=source_hdus,
        target_wcs=target_wcs,
        shape_out=shape_out,
        device=device,
        num_threads=num_threads,
        requires_grad=requires_grad,
        conserve_flux=conserve_flux,
        compute_jacobian=compute_jacobian,
    )

    order = validate_interpolation_order(order)

    # Choose between chunked and non-chunked processing
    if max_memory_mb is not None:
        result = reproject_chunked(
            reprojection,
            max_memory_mb=max_memory_mb,
            safety_factor=chunk_safety_factor,
            interpolation_mode=order,
            show_progress=show_chunk_progress,
        ).squeeze(0)
    else:
        result = reprojection.interpolate_source_image(interpolation_mode=order)

    # Convert output format
    if requires_grad:
        result = result.cpu()
    else:
        result = result.cpu().numpy().astype(np.float32)

    torch.cuda.empty_cache()

    return result
