from typing import List, Tuple, Union

import numpy as np
import torch
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS

from .sip import (
    apply_inverse_sip_distortion,
    apply_sip_distortion,
    get_sip_coeffs
)
from .utils import get_device

EPSILON = 1e-10


# Helper functions for trigonometric calculations
def atan2d(y, x):
    """PyTorch implementation of WCSLib's atan2d function"""
    return torch.rad2deg(torch.atan2(y, x))


def sincosd(angle_deg):
    """PyTorch implementation of WCSLib's sincosd function"""
    angle_rad = torch.deg2rad(angle_deg)
    return torch.sin(angle_rad), torch.cos(angle_rad)

@torch.jit.script
def interpolate_image(
    source_image: torch.Tensor, grid: torch.Tensor, interpolation_mode: str
) -> torch.Tensor:
    """JIT-compiled image interpolation using grid_sample"""
    return torch.nn.functional.grid_sample(
        source_image,
        grid,
        mode=interpolation_mode,
        align_corners=True,
        padding_mode="zeros",
    )


class Reproject:
    def __init__(
        self, source_hdus: List[PrimaryHDU], target_wcs: WCS, shape_out: Tuple[int, int]
    ):
        """
        Initialize a reprojection operation between source and target image frames.

        This constructor sets up the necessary components for reprojecting an astronomical
        image from one World Coordinate System (WCS) to another. It stores the source
        and target WCS information, images, and creates a coordinate grid for the target
        image that will be used in the reprojection process.

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the source image

        target_wcs : WCS
            WCS for the target in an astropy.wcs compatible format.

        shape_out: Tuple[int, int]
            Shape of the output image

        Notes
        -----
        This constructor creates a coordinate grid spanning the entire target image,
        which will be used for the pixel-to-world and world-to-pixel transformations
        during reprojection. The grid is created with 'ij' indexing, where the first
        dimension corresponds to y (rows) and the second to x (columns).

        The coordinate grid is stored as a tuple of tensors (batch, y_grid, x_grid), where
        each element has the same shape as the target image.
        Examples
        --------
        >>> # Initialize the reprojection object
        >>> reproject = Reproject(source_hdus, target_wcs)

        """
        # Set device
        self.device = get_device()

        self.batch_source_images = self._prepare_source_images(source_hdus)
        # Initialize the WCS objects
        self.batch_source_wcs_params = self._prepare_batch_wcs_params(source_hdus)
        self.target_wcs_params = self._extract_wcs_params(target_wcs)

        # Define target grid
        self.target_grid = self._create_batch_target_grid(shape_out)

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
            source_images = [
                torch.tensor(hdu.data, dtype=torch.float64, device=self.device)
                for hdu in source_hdus
            ]
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

    def _prepare_batch_wcs_params(self, source_hdus: List[PrimaryHDU]) -> List[dict]:
        """
        Prepare batch of WCS parameters

        Parameters
        ----------
        source_hdus : List[PrimaryHDU]
            List of HDUs containing the data and the header information for the source image
        """
        return [self._extract_wcs_params(WCS(hdu.header)) for hdu in source_hdus]

    def _create_batch_target_grid(self, shape_out: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # Batch matrix multiplication for scaling
        # Reshape to [B, H*W, 2] for batch matrix multiplication
        pixel_offsets = torch.stack([u.reshape(B, -1), v.reshape(B, -1)], dim=-1)
        transformed = torch.bmm(
            pixel_offsets, CD_matrix.T.unsqueeze(0).expand(B, -1, -1)
        )

        x_scaled = transformed[:, :, 0].reshape(B, H, W)
        y_scaled = transformed[:, :, 1].reshape(B, H, W)

        # Compute radial distance
        r = torch.sqrt(x_scaled**2 + y_scaled**2)
        r0 = torch.tensor(180.0 / torch.pi, device=self.device)

        # Compute phi
        phi = torch.zeros_like(r)
        non_zero_r = r != 0
        phi[non_zero_r] = torch.rad2deg(
            torch.atan2(-x_scaled[non_zero_r], y_scaled[non_zero_r])
        )

        # Compute theta
        theta = torch.rad2deg(torch.atan2(r0, r))

        # Convert to radians
        phi_rad = torch.deg2rad(phi)
        theta_rad = torch.deg2rad(theta)
        ra0_rad = crval[0] * torch.pi / 180.0
        dec0_rad = crval[1] * torch.pi / 180.0

        # Spherical coordinate calculations
        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)
        sin_dec0 = torch.sin(dec0_rad)
        cos_dec0 = torch.cos(dec0_rad)

        sin_dec = sin_theta * sin_dec0 + cos_theta * cos_dec0 * cos_phi
        dec_rad = torch.arcsin(sin_dec)

        y_term = cos_theta * sin_phi
        x_term = sin_theta * cos_dec0 - cos_theta * sin_dec0 * cos_phi
        ra_rad = ra0_rad + torch.atan2(-y_term, x_term)

        # Convert to degrees and normalize
        ra = torch.rad2deg(ra_rad) % 360.0
        dec = torch.rad2deg(dec_rad)

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
            ra0_rad = crval[0] * torch.pi / 180.0
            dec0_rad = crval[1] * torch.pi / 180.0
            # Convert numpy arrays to torch tensors if needed
            if not isinstance(ra, torch.Tensor):
                ra = torch.tensor(ra, device=self.device)
                dec = torch.tensor(dec, device=self.device)
            # Step 1: Convert from world to native spherical coordinates
            # Calculate the difference in RA
            delta_ra = ra_rad - ra0_rad
            # Calculate sine and cosine values
            sin_dec = torch.sin(dec_rad)
            cos_dec = torch.cos(dec_rad)
            sin_dec0 = torch.sin(dec0_rad)
            cos_dec0 = torch.cos(dec0_rad)
            sin_delta_ra = torch.sin(delta_ra)
            cos_delta_ra = torch.cos(delta_ra)
            # Calculate the native spherical coordinates using the correct sign conventions
            y_phi = -cos_dec * sin_delta_ra  # Note the negative sign
            # Calculate the denominator for phi
            x_phi = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_delta_ra
            # Calculate native longitude (phi)
            phi = atan2d(y_phi, x_phi)
            # Calculate native latitude (theta)
            theta = torch.rad2deg(
                torch.arcsin(sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_delta_ra)
            )
            # Step 2: Apply the TAN projection (tans2x function from WCSLib)
            # Calculate sine and cosine of phi and theta
            sin_phi, cos_phi = sincosd(phi)
            sin_theta, cos_theta = sincosd(theta)
            # Check for singularity (when sin_theta is zero)
            eps = 1e-10
            if torch.any(torch.abs(sin_theta) < eps):
                raise ValueError("Singularity in tans2x: theta close to 0 degrees")
            # r0 is the radius scaling factor (typically 180.0/π)
            r0 = torch.tensor(180.0 / torch.pi, device=self.device)
            # Calculate the scaling factor r with correct sign
            r = r0 * cos_theta / sin_theta
            # Calculate intermediate world coordinates (x_scaled, y_scaled)
            # With the corrected signs based on your findings
            x_scaled = -r * sin_phi  # Note the negative sign
            y_scaled = r * cos_phi
            # Step 3: Apply the inverse of the CD matrix to get pixel offsets
            # First, construct the CD matrix
            CD_matrix = pc_matrix * cdelt
            # Calculate the inverse of the CD matrix
            CD_inv = torch.linalg.inv(CD_matrix)
            # Handle batch processing for arrays
            if ra.dim() == 0:  # scalar inputs
                standard_coords = torch.tensor(
                    [x_scaled.item(), y_scaled.item()],
                    device=self.device,
                    dtype=torch.float64,
                )
                pixel_offsets = torch.matmul(CD_inv, standard_coords)
                u = pixel_offsets[0]
                v = pixel_offsets[1]
            else:  # array inputs
                # Reshape for batch processing if needed
                if ra.dim() > 1:
                    original_shape = ra.shape
                    x_scaled_flat = x_scaled.reshape(-1)
                    y_scaled_flat = y_scaled.reshape(-1)
                else:
                    x_scaled_flat = x_scaled
                    y_scaled_flat = y_scaled
                # Stack for batch matrix multiplication
                standard_coords = torch.stack(
                    [x_scaled_flat, y_scaled_flat], dim=1
                )  # Shape: [N, 2]
                # Use batch matrix multiplication
                pixel_offsets = torch.matmul(standard_coords, CD_inv.T)  # Shape: [N, 2]
                u = pixel_offsets[:, 0]
                v = pixel_offsets[:, 1]
                # Reshape back to original dimensions if needed
                if ra.dim() > 1:
                    u = u.reshape(original_shape)
                    v = v.reshape(original_shape)
            if sip_coeffs is not None:
                u, v = apply_inverse_sip_distortion(u, v, sip_coeffs, self.device)
            # Step 4: Add the reference pixel to get final pixel coordinates
            # Remember to add (CRPIX-1) to account for 1-based indexing in FITS/WCS
            x_pixel = u + (crpix[0] - 1)
            y_pixel = v + (crpix[1] - 1)
            batch_x_pixel[b] = x_pixel
            batch_y_pixel[b] = y_pixel

        return batch_x_pixel, batch_y_pixel

    def interpolate_source_image(self, interpolation_mode="bilinear") -> torch.Tensor:
        """
        Interpolate the source image at the calculated source coordinates with flux conservation.

        This method performs the actual pixel resampling needed for reprojection
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

        1. Local flux conservation: The image and a "ones" tensor are interpolated
           together, and the interpolated image is divided by the interpolated ones
           tensor (footprint) to correct for any flux spreading during interpolation.

        Areas in the target image that map outside the source image boundaries
        will be filled with zeros (using 'zeros' padding_mode).

        This method is particularly suitable for high-precision photometry with
        extended sources, as it properly preserves both the background noise
        characteristics and the flux distribution of sources.
        """
        # Get source coordinates
        x_source, y_source = self.calculate_sourceCoords()
        B, H, W = self.batch_source_images.shape
        # Normalize coordinates
        x_normalized = 2.0 * (x_source / (W - 1)) - 1.0
        y_normalized = 2.0 * (y_source / (H - 1)) - 1.0
        # Create sampling grid
        grid = torch.stack([x_normalized, y_normalized], dim=-1)
        # Prepare images and grid for grid_sample
        source_images = self.batch_source_images.unsqueeze(1)  # [B, 1, H, W]
        ones = torch.ones_like(source_images)
        # Combine images with ones for footprint calculation
        combined = torch.cat([source_images, ones], dim=1)  # [B, 2, H, W]
        # Single grid_sample call
        combined_result = interpolate_image(combined, grid, interpolation_mode)
        # Split the results
        resampled_image = combined_result[:, 0].squeeze()
        resampled_footprint = combined_result[:, 1].squeeze()
        # Create output array initialized with zeros
        result = torch.full_like(resampled_image, torch.nan)
        # Apply footprint correction only where footprint is significant
        valid_pixels = resampled_footprint > EPSILON
        # Apply footprint correction only where footprint is significant
        if torch.any(valid_pixels):
            # Normalize by the footprint where valid
            result[valid_pixels] = (
                resampled_image[valid_pixels] / resampled_footprint[valid_pixels]
            )
        else:
            print("WARNING: No valid pixels found in footprint!")
            # FALLBACK: Use raw interpolated values without footprint correction
            print("Fallback: Using raw interpolated values")
            result = resampled_image

        return result


def calculate_reprojection(
    source_hdus: Union[PrimaryHDU, List[PrimaryHDU]],
    target_wcs: WCS,
    shape_out: Tuple[int, int],
    interpolation_mode="nearest",
):
    """
    Reproject an astronomical image from a source WCS to a target WCS.

    This high-level function provides a convenient interface for image reprojection,
    handling all the necessary steps: WCS extraction, tensor creation, and interpolation.
    It converts FITS HDU objects to the internal representation, performs the reprojection,
    and returns the resulting image as a PyTorch tensor.

    Parameters
    ----------
    source_hdus : Union[PrimaryHDU, List[PrimaryHDU]]
        The source image HDU list containing the image data to be reprojected and
        its associated WCS information in the header.

    target_wcs : WCS
        WCS information fopr the target.

    shape_out: Tuple[int, int]
        Shape of the resampled array

    interpolation_mode : str, default 'nearest'
        The interpolation method to use when resampling the source image.
        Options:
        - 'nearest' : Nearest neighbor interpolation (fastest, default)
        - 'bilinear' : Bilinear interpolation (good balance of speed/quality)
        - 'bicubic' : Bicubic interpolation (highest quality, slowest)

    Returns
    -------
    numpy.ndarray
        The reprojected image as a numpy ndarray with the same shape as
        target_hdu.data.

    Notes
    -----
    This function automatically:
    - Detects and uses GPU acceleration if available
    - Handles byte order conversion for tensor creation
    - Converts data to float32 for processing
    - Creates WCSHeader objects from FITS headers

    To save the result as a FITS file, you will need to convert the tensor
    back to a NumPy array and create a new FITS HDU with the target WCS header.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from astropy.wcs import WCS
    >>> import torch
    >>> from reprojection import calculate_reprojection
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
    ...     shape_out=target_hdu[0].data.shape,
    ...     interpolation_mode='bilinear'
    ... )
    >>>
    >>> # Convert back to NumPy and save as FITS
    >>> reprojected_np = reprojected.cpu().numpy()
    >>> output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
    >>> output_hdu.writeto('reprojected_image.fits', overwrite=True)
    """
    # Convert single HDU to list if not already a list
    if not isinstance(source_hdus, list):
        source_hdus = [source_hdus]
    reprojection = Reproject(
        source_hdus=source_hdus, target_wcs=target_wcs, shape_out=shape_out
    )
    result = reprojection.interpolate_source_image(interpolation_mode=interpolation_mode).cpu().numpy()
    torch.cuda.empty_cache()
    return result
