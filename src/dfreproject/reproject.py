import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from astropy.io.fits import Header, PrimaryHDU
from astropy.wcs import WCS

from .sip import apply_inverse_sip_distortion, apply_sip_distortion, get_sip_coeffs
from .tensorhdu import TensorHDU
from .utils import get_device

logger = logging.getLogger(__name__)

EPSILON = 1e-10
VALID_ORDERS = ["bicubic", "bilinear", "nearest", "nearest-neighbors"]


def validate_interpolation_order(order: str) -> str:
    """
    Function to validate the requested interpolation order.

    The order must be one of the following: {valid_orders}

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
    PyTorch implementation of WCSLib's atan2d function

    Parameters
    ----------
    y : torch.Tensor
        y coordinate(s)
    x : torch.Tensor
        x coordinate(s)

    Returns
    -------
    torch.Tensor
        atan2d(y, x) in degrees
    """
    return torch.rad2deg(torch.atan2(y, x))


def sincosd(angle_deg: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of WCSLib's sincosd function

    Parameters
    ----------
    angle_deg : torch.Tensor
        angle in degrees

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor)
        sin(angle) in degrees, cos(angle) in degrees
    """
    angle_rad = torch.deg2rad(angle_deg)
    return torch.sin(angle_rad), torch.cos(angle_rad)


@torch.jit.script
def interpolate_image(
    source_image: torch.Tensor, grid: torch.Tensor, interpolation_mode: str
) -> torch.Tensor:
    """
    JIT-compiled image interpolation using grid_sample

    Parameters
    ----------
    source_image : torch.Tensor
        Source image to interpolate
    grid : torch.Tensor
        Grid on which to interpolate
    interpolation_mode: str
        Interpolation mode to use
    """
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
        conserve_flux: bool = False,
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
            List of HDUs containing the data and the header information for the source image

        target_wcs : WCS
            WCS for the target in an astropy.wcs compatible format.

        shape_out: Tuple[int, int]
            Shape of the output image

        device: str
            Device to use for computations. Defaults to GPU if available otherwise uses CPU

        num_threads: int
            Number of threads to use on CPU

        conserve_flux: bool, optional
            if True, enables flux conservation through Jacobian calculation. Note that this increases RAM usage.

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

        # Define target grid
        self.target_grid = self._create_batch_target_grid(shape_out)
        self.conserve_flux = conserve_flux

    def _prepare_source_images(self, source_hdus: List[PrimaryHDU]) -> torch.Tensor:
        """
        Prepare batch of source images as a single tensor


        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the source image

        Returns
        -------
        source_image : torch.Tensor
            Stack of source image tensors
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
        Extract key WCS parameters into a dictionary for efficient tensor operations

        Returns a dictionary with pre-computed tensor parameters

        Parameters
        ----------
        wcs : WCS
            WCS information

        Returns
        -------
        wcs_params : dict
            WCS parameters

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
        Prepare batch of WCS parameters

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the source image

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
        Create a batched target grid matching the number of source images

        Parameters
        ----------
        shape_out : Tuple[int, int]
            Shape of the output image
        """
        B = len(self.batch_source_images)
        H, W = shape_out

        # Create meshgrid and repeat for batch
        y_grid = (
            torch.arange(H, dtype=torch.float64, device=self.device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_grid = (
            torch.arange(W, dtype=torch.float64, device=self.device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

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

        These steps use the target wcs parameters

        Parameters
        ----------
        x_grid : torch.Tensor, optional
            Batch of x-coordinates. If None, uses target grid x-coordinates
        y_grid : torch.Tensor, optional
            Batch of y-coordinates. If None, uses target grid y-coordinates

        Returns
        -------
        tuple
            Batched RA and Dec coordinates
        """

        if x_grid is None or y_grid is None:
            y_grid, x_grid = self.target_grid
        else:
            # Ensure x and y are tensors on the correct device
            if not isinstance(x_grid, torch.Tensor):
                x_grid = torch.tensor(x_grid, dtype=torch.float64, device=self.device)
            if not isinstance(y_grid, torch.Tensor):
                y_grid = torch.tensor(y_grid, dtype=torch.float64, device=self.device)

            # Ensure consistent batch dimensions
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

        # Compute pixel offsets
        u = x_grid - (crpix[0] - 1)
        v = y_grid - (crpix[1] - 1)

        # Apply SIP distortion if present
        if sip_coeffs is not None:
            u, v = apply_sip_distortion(u, v, sip_coeffs, self.device)

        # Apply PC Matrix and CDELT
        CD_matrix = pc_matrix * cdelt
        del pc_matrix, cdelt

        # Batch matrix multiplication for scaling
        # Reshape to [B, H*W, 2] for batch matrix multiplication
        pixel_offsets = torch.stack([u.reshape(B, -1), v.reshape(B, -1)], dim=-1)
        transformed = torch.bmm(
            pixel_offsets, CD_matrix.T.unsqueeze(0).expand(B, -1, -1)
        )
        del pixel_offsets, CD_matrix, u, v
        x_scaled = transformed[:, :, 0].reshape(B, H, W)
        y_scaled = transformed[:, :, 1].reshape(B, H, W)
        del transformed
        # Compute radial distance
        r = torch.sqrt(x_scaled**2 + y_scaled**2)
        r0 = torch.tensor(180.0 / torch.pi, device=self.device)

        # Compute phi
        phi = torch.zeros_like(r)
        non_zero_r = r != 0
        phi[non_zero_r] = torch.rad2deg(
            torch.atan2(-x_scaled[non_zero_r], y_scaled[non_zero_r])
        )
        del x_scaled, y_scaled, non_zero_r
        # Convert to radians
        phi_rad = torch.deg2rad(phi)
        del phi
        theta_rad = torch.deg2rad(torch.rad2deg(torch.atan2(r0, r)))
        del r0, r
        ra0_rad = crval[0] * torch.pi / 180.0
        dec0_rad = crval[1] * torch.pi / 180.0

        sin_dec = torch.sin(theta_rad) * torch.sin(dec0_rad) + torch.cos(
            theta_rad
        ) * torch.cos(dec0_rad) * torch.cos(phi_rad)
        dec_rad = torch.arcsin(sin_dec)
        del sin_dec
        # y_term = torch.cos(theta_rad) * torch.sin(phi_rad)
        # x_term = torch.sin(theta_rad) * torch.cos(dec0_rad) - torch.cos(theta_rad) * torch.sin(dec0_rad) * torch.cos(phi_rad)
        ra_rad = ra0_rad + torch.atan2(
            -torch.cos(theta_rad) * torch.sin(phi_rad),
            torch.sin(theta_rad) * torch.cos(dec0_rad)
            - torch.cos(theta_rad) * torch.sin(dec0_rad) * torch.cos(phi_rad),
        )
        del theta_rad, phi_rad, ra0_rad, dec0_rad
        # Convert to degrees and normalize
        ra = torch.rad2deg(ra_rad) % 360.0
        dec = torch.rad2deg(dec_rad)
        del ra_rad, dec_rad
        return ra, dec

    def calculate_sourceCoords(self):
        """
        Calculate source image pixel coordinates corresponding to each target image pixel.

        This function repeats the same steps in self.calculate_skyCoords() except in the opposite order and with the source coordinate wcs.

        Returns
        -------
        torch.Tensor
            Batch of source image pixel coordinates
        """
        # Get batch of sky coordinates
        batch_ra, batch_dec = self.calculate_skyCoords()
        B, H, W = batch_ra.shape

        # Prepare to collect results for each source image
        batch_x_pixel = torch.zeros((B, H, W), dtype=torch.float64, device=self.device)
        batch_y_pixel = torch.zeros((B, H, W), dtype=torch.float64, device=self.device)

        # Process each source image's coordinates
        for b in range(B):
            # Get WCS parameters for this specific source image
            source_wcs_params = self.batch_source_wcs_params[b]
            # Unpack WCS parameters
            crpix = source_wcs_params["crpix"]
            crval = source_wcs_params["crval"]
            pc_matrix = source_wcs_params["pc_matrix"]
            cdelt = source_wcs_params["cdelt"]
            sip_coeffs = source_wcs_params["sip_coeffs"]
            # Extract this batch's RA and Dec
            ra = batch_ra[b]
            dec = batch_dec[b]
            # Conversion calculations
            ra_rad = torch.deg2rad(ra)
            dec_rad = torch.deg2rad(dec)
            ra_dim = ra.dim()
            ra_shape = ra.shape
            del ra, dec
            ra0_rad = crval[0] * torch.pi / 180.0
            dec0_rad = crval[1] * torch.pi / 180.0
            # Convert numpy arrays to torch tensors if needed
            # if not isinstance(ra, torch.Tensor):
            #     ra = torch.tensor(ra, device=self.device)
            # Step 1: Convert from world to native spherical coordinates
            # Calculate the native spherical coordinates using the correct sign conventions
            y_phi = -torch.cos(dec_rad) * torch.sin(
                ra_rad - ra0_rad
            )  # Note the negative sign
            # Calculate the denominator for phi
            x_phi = torch.sin(dec_rad) * torch.cos(dec0_rad) - torch.cos(
                dec_rad
            ) * torch.sin(dec0_rad) * torch.cos(ra_rad - ra0_rad)
            # Calculate native longitude (phi)
            phi = atan2d(y_phi, x_phi)
            del x_phi, y_phi
            # Calculate native latitude (theta)
            theta = torch.rad2deg(
                torch.arcsin(
                    torch.sin(dec_rad) * torch.sin(dec0_rad)
                    + torch.cos(dec_rad)
                    * torch.cos(dec0_rad)
                    * torch.cos(ra_rad - ra0_rad)
                )
            )
            del ra_rad, dec_rad
            # del sin_dec, cos_dec, sin_dec0, cos_dec0, cos_delta_ra
            # Step 2: Apply the TAN projection (tans2x function from WCSLib)
            # Calculate sine and cosine of phi and theta
            sin_phi, cos_phi = sincosd(phi)
            del phi
            sin_theta, cos_theta = sincosd(theta)
            del theta
            # Check for singularity (when sin_theta is zero)
            eps = 1e-10
            if torch.any(torch.abs(sin_theta) < eps):
                raise ValueError("Singularity in tans2x: theta close to 0 degrees")
            # r0 is the radius scaling factor (typically 180.0/π)
            r0 = torch.tensor(180.0 / torch.pi, device=self.device)
            # Calculate the scaling factor r with correct sign
            r = r0 * cos_theta / sin_theta
            del cos_theta, sin_theta, r0
            # Calculate intermediate world coordinates (x_scaled, y_scaled)
            # With the corrected signs based on your findings
            x_scaled = -r * sin_phi  # Note the negative sign
            y_scaled = r * cos_phi
            del sin_phi, cos_phi, r
            # Step 3: Apply the inverse of the CD matrix to get pixel offsets
            # First, construct the CD matrix
            CD_matrix = pc_matrix * cdelt
            del pc_matrix, cdelt
            # Calculate the inverse of the CD matrix
            CD_inv = torch.linalg.inv(CD_matrix)
            del CD_matrix
            # Handle batch processing for arrays
            if ra_dim == 0:  # scalar inputs
                standard_coords = torch.tensor(
                    [x_scaled.item(), y_scaled.item()],
                    device=self.device,
                    dtype=torch.float64,
                )
                pixel_offsets = torch.matmul(CD_inv, standard_coords)
                u = pixel_offsets[0]
                v = pixel_offsets[1]
                del CD_inv, pixel_offsets, standard_coords
            else:  # array inputs
                # Reshape for batch processing if needed
                if ra_dim > 1:
                    original_shape = ra_shape
                    x_scaled_flat = x_scaled.reshape(-1)
                    y_scaled_flat = y_scaled.reshape(-1)
                else:
                    x_scaled_flat = x_scaled
                    y_scaled_flat = y_scaled
                # Stack for batch matrix multiplication
                del x_scaled, y_scaled
                standard_coords = torch.stack(
                    [x_scaled_flat, y_scaled_flat], dim=1
                )  # Shape: [N, 2]
                del x_scaled_flat, y_scaled_flat
                # Use batch matrix multiplication
                pixel_offsets = torch.matmul(standard_coords, CD_inv.T)  # Shape: [N, 2]
                u = pixel_offsets[:, 0]
                v = pixel_offsets[:, 1]
                del CD_inv, pixel_offsets, standard_coords
                # Reshape back to original dimensions if needed
                if ra_dim > 1:
                    u = u.reshape(original_shape)
                    v = v.reshape(original_shape)
            if sip_coeffs is not None:
                u, v = apply_inverse_sip_distortion(u, v, sip_coeffs, self.device)

            # Step 4: Add the reference pixel to get final pixel coordinates
            # Remember to add (CRPIX-1) to account for 1-based indexing in FITS/WCS
            x_pixel = u + (crpix[0] - 1)
            y_pixel = v + (crpix[1] - 1)
            del u, v
            batch_x_pixel[b] = x_pixel
            batch_y_pixel[b] = y_pixel
            del x_pixel, y_pixel, crpix, crval
        del batch_ra, batch_dec
        return batch_x_pixel, batch_y_pixel

    def calculate_jacobian_determinant_sparse(self, downsample_factor=4):
        """
        Calculate Jacobian determinant using sparse sampling + interpolation.
        Much faster for large grids since WCS transformations are smooth.
        Memory optimized with explicit cleanup and torch.no_grad().
        """
        # Disable gradient computation for memory efficiency
        with torch.no_grad():
            eps = 1e-4
            target_y_orig, target_x_orig = self.target_grid
            orig_shape = target_y_orig.shape

            # Force garbage collection before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Handle both 2D and 3D tensors
        if len(orig_shape) == 3:
            # 3D case: [batch, height, width] or similar
            sparse_y = target_y_orig[:, ::downsample_factor, ::downsample_factor]
            sparse_x = target_x_orig[:, ::downsample_factor, ::downsample_factor]
            spatial_dims = orig_shape[1:]  # Height, width for interpolation
        else:
            # 2D case: [height, width]
            sparse_y = target_y_orig[::downsample_factor, ::downsample_factor]
            sparse_x = target_x_orig[::downsample_factor, ::downsample_factor]
            spatial_dims = orig_shape  # Full shape for interpolation

        # Compute Jacobian on sparse grid using your original method logic
        self.target_grid = (sparse_y, sparse_x)

        try:
            # Get original coordinates on sparse grid
            x_source, y_source = self.calculate_sourceCoords()

            # Perturbation in target x-direction
            target_x_plus = sparse_x + eps
            self.target_grid = (sparse_y, target_x_plus)
            x_source_dx, y_source_dx = self.calculate_sourceCoords()
            dx_source_dx = (x_source_dx - x_source) / eps
            dy_source_dx = (y_source_dx - y_source) / eps
            del x_source_dx, y_source_dx, target_x_plus

            # Perturbation in target y-direction
            target_y_plus = sparse_y + eps
            self.target_grid = (target_y_plus, sparse_x)
            x_source_dy, y_source_dy = self.calculate_sourceCoords()
            dx_source_dy = (x_source_dy - x_source) / eps
            dy_source_dy = (y_source_dy - y_source) / eps
            del x_source_dy, y_source_dy, target_y_plus, x_source, y_source

            # Compute Jacobian determinant (target→source)
            jacobian_det_forward = (
                dx_source_dx * dy_source_dy - dx_source_dy * dy_source_dx
            )
            del dx_source_dx, dy_source_dx, dx_source_dy, dy_source_dy

            # For flux conservation, we need the inverse (source→target)
            jacobian_sparse = 1.0 / torch.abs(jacobian_det_forward)
            del jacobian_det_forward

            # Handle numerical issues
            jacobian_sparse = torch.where(
                torch.isfinite(jacobian_sparse) & (jacobian_sparse > 1e-10),
                jacobian_sparse,
                torch.ones_like(jacobian_sparse),
            )
            jacobian_sparse = torch.clamp(jacobian_sparse, min=1e-6, max=1e6)

        finally:
            # Restore original grid
            self.target_grid = (target_y_orig, target_x_orig)

        # Interpolate back to full resolution
        if len(orig_shape) == 3:
            # 3D case: interpolate each item in the batch
            jacobian_results = []
            for i in range(orig_shape[0]):
                # Extract 2D slice from sparse result
                if jacobian_sparse.dim() == 3:
                    sparse_2d = jacobian_sparse[i]
                else:
                    # If sparse result collapsed to 2D, use it directly
                    sparse_2d = jacobian_sparse

                # Interpolate this slice
                interp_2d = torch.nn.functional.interpolate(
                    sparse_2d.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
                    size=spatial_dims,  # Target spatial dimensions
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()  # Remove batch + channel dims

                jacobian_results.append(interp_2d.unsqueeze(0))  # Add batch dim

            jacobian_det = torch.cat(jacobian_results, dim=0)

        else:
            # 2D case: direct interpolation
            jacobian_det = torch.nn.functional.interpolate(
                jacobian_sparse.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
                size=spatial_dims,
                mode="bilinear",
                align_corners=True,
            ).squeeze()  # Remove batch + channel dims

        return jacobian_det

    def interpolate_source_image(self, interpolation_mode="bilinear") -> torch.Tensor:
        """
        Interpolate the source image at the calculated source coordinates with flux conservation.

        This method performs the actual pixel resampling needed for dfreproject
        while preserving the total flux (photometric accuracy). It implements a
        footprint-based approach similar to that used in reproject_interp from the
        Astropy package.

        The method uses a combined tensor approach for computational efficiency,
        performing both image resampling and footprint tracking in a single operation.
        Total flux is conserved locally (via footprint correction).

        Parameters
        ----------
        interpolation_mode : str, default 'bilinear'
            The interpolation mode to use when sampling the source image.
            Options include:
            - 'nearest' : Nearest neighbor interpolation (no interpolation)
            - 'bilinear' : Bilinear interpolation (default)
            - 'bicubic' : Bicubic interpolation

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
        2. Jacobian correction for full flux conservation: Multiply the footprint-corrected flux
            by the determinant of the Jacobian to handle changes in area during the reprojection

        The Jacobian correction can be circumvented if you set conserve_flux=False. However, the default behavior is to
        include this.

        Areas in the target image that map outside the source image boundaries
        will be filled with zeros (using 'zeros' padding_mode).

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
        # Create output array initialized with zeros
        result = torch.full_like(combined_result[:, 0].squeeze(), torch.nan)
        # Apply footprint correction only where footprint is significant
        valid_pixels = combined_result[:, 1].squeeze() > EPSILON
        # Apply footprint correction only where footprint is significant
        if torch.any(valid_pixels):
            # Normalize by the footprint where valid
            result[valid_pixels] = (
                combined_result[:, 0].squeeze()[valid_pixels]
                / combined_result[:, 1].squeeze()[valid_pixels]
            )
            if self.conserve_flux:  # Include Jacobian determinant computation
                jacobian_det = self.calculate_jacobian_determinant_sparse(
                    downsample_factor=2
                ).squeeze()
                result[valid_pixels] = result[valid_pixels] / jacobian_det[valid_pixels]
        else:
            result = combined_result[:, 0].squeeze() / combined_result[:, 1].squeeze()
            logger.warning(
                "No valid pixels found in footprint! Using raw interpolated values"
            )
        del valid_pixels

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
    converve_flux: bool = False,
):
    """
    Reproject an astronomical image from a source WCS to a target WCS.

    This high-level function provides a convenient interface for image reprojection,
    handling all the necessary steps: WCS extraction, tensor creation, and interpolation.
    It converts FITS HDU objects to the internal representation, performs the reprojection,
    and returns the resulting image as a NumPy array or PyTorch tensor.

    Parameters
    ----------
    source_hdus : Union[
        PrimaryHDU,
        TensorHDU,
        Tuple[np.ndarray, Union[WCS, Header]],
        Tuple[torch.Tensor, Union[WCS, Header]],
        List[Union[PrimaryHDU, Tuple[np.ndarray, Union[WCS, Header]]]]
    ]
        The source image(s) to be reprojected, which can be a PrimaryHDU, TensorHDU,
        a tuple of (data, WCS/Header), or a list of such items.

    target_wcs : Union[WCS, Header]
        WCS information for the target. If a Header is passed it will be converted to WCS.

    shape_out: Optional[Tuple[int, int]]
        Shape of the resampled array. If not provided, the output shape will match the input.

    order : str, default 'nearest'
        The interpolation method to use when resampling the source image.
        Options:
        - 'nearest' : Nearest neighbor interpolation (fastest, default)
        - 'bilinear' : Bilinear interpolation (good balance of speed/quality)
        - 'bicubic' : Bicubic interpolation (highest quality, slowest)

    device: str, optional
        Device to use for computations. Defaults to GPU if available, otherwise uses CPU.

    num_threads: int, optional
        Number of threads to use on CPU.

    requires_grad: bool, optional
        If True, enables autograd for PyTorch tensors.

    conserve_flux: bool, optional
        if True, enables flux conservation through Jacobian calculation. Note that this increases RAM usage.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The reprojected image as a numpy ndarray (default) or PyTorch tensor if requires_grad=True.

    Notes
    -----
    This function automatically:
    - Detects and uses GPU acceleration if available
    - Handles byte order conversion for tensor creation
    - Converts data to float64 for processing
    - Converts Header to WCS if needed

    To save the result as a FITS file, convert the tensor
    back to a NumPy array and create a new FITS HDU with the target WCS header.

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
                header = wcs_or_header.to_header()
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
        conserve_flux=converve_flux,
    )
    order = validate_interpolation_order(order)

    if requires_grad:
        result = reprojection.interpolate_source_image(interpolation_mode=order).cpu()
    else:
        result = (
            reprojection.interpolate_source_image(interpolation_mode=order)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    torch.cuda.empty_cache()
    return result
