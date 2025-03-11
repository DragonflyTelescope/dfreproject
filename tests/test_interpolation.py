import pytest
import torch
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits


# Import the actual Reproject class - adjust as needed
# from dfreproject.reproject import Reproject

@pytest.mark.integration
class TestInterpolationIntegration:
    """Integration tests for the interpolate_source_image method."""

    @pytest.fixture
    def setup_real_projection(self, source_fits_file, target_fits_file, device):
        """Set up a real projection with actual FITS files."""
        # Load the HDUs
        source_hdu = fits.open(source_fits_file)[0]
        target_hdu = fits.open(target_fits_file)[0]
        # Convert data to native byte order if needed
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        if hasattr(target_hdu, 'data') and target_hdu.data is not None:
            target_hdu.data = np.asarray(target_hdu.data, dtype=np.float64).copy()
        # Import the Reproject class
        from dfreproject.reproject import Reproject

        # Create actual instance
        reproject = Reproject([source_hdu], WCS(target_hdu.header), shape_out=target_hdu.data.shape)

        # Set device if needed
        if hasattr(reproject, 'set_device'):
            reproject.set_device(device)

        return reproject

    def test_flux_conservation(self, setup_real_projection):
        """Test that flux is approximately conserved during interpolation."""
        reproject = setup_real_projection

        # Get original flux
        original_flux = reproject.batch_source_images[0].sum().item()

        # Perform interpolation
        result = reproject.interpolate_source_image(interpolation_mode="bilinear")

        # Get interpolated flux
        interpolated_flux = result.nan_to_num(0).sum().item()

        # Flux should be approximately conserved
        # Allow some tolerance for edge effects and numerical precision
        # Actual tolerance may need adjustment based on your specific transformations
        assert abs(interpolated_flux - original_flux) / original_flux < 0.2

    def test_multiple_interpolation_modes(self, setup_real_projection):
        """Test that all interpolation modes produce valid results."""
        reproject = setup_real_projection

        # Test all interpolation modes
        modes = ["nearest", "bilinear", "bicubic"]

        for mode in modes:
            # Perform interpolation
            result = reproject.interpolate_source_image(interpolation_mode=mode)

            # Basic shape check
            assert result.unsqueeze(0).shape == reproject.target_grid[0].shape

    def test_large_rotation(self, device):
        """Test a large rotation transformation."""
        # Create custom source and target WCS with a 90-degree rotation
        from astropy.wcs import WCS

        # Create source WCS
        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.cdelt = [0.1, 0.1]
        source_wcs.wcs.crval = [0, 0]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create target WCS with 90-degree rotation
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [50, 50]
        target_wcs.wcs.cdelt = [0.1, 0.1]
        target_wcs.wcs.crval = [0, 0]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # Apply 90-degree rotation
        target_wcs.wcs.pc = [[0, -1], [1, 0]]

        # Create source with a recognizable pattern
        source_data = np.zeros((100, 100))
        # Add a horizontal line
        source_data[50, 20:80] = 100.0

        source_hdu = fits.PrimaryHDU(data=source_data.astype(np.float32))
        source_hdu.header.update(source_wcs.to_header())

        target_data = np.zeros((100, 100), dtype=np.float32)
        target_hdu = fits.PrimaryHDU(data=target_data)
        target_hdu.header.update(target_wcs.to_header())

        # Import the Reproject class
        from dfreproject.reproject import Reproject

        # Create reproject instance
        reproject = Reproject([source_hdu], target_wcs, shape_out=target_data.shape)

        # Set device if needed
        if hasattr(reproject, 'set_device'):
            reproject.set_device(device)

        # Perform interpolation
        result = reproject.interpolate_source_image(interpolation_mode="bilinear")
        result_np = result.cpu().numpy()

        # After 90-degree rotation, the horizontal line should become vertical
        # Check for high values in the vertical direction
        vertical_line_values = result_np[20:80, 50]
        horizontal_line_values = result_np[50, 20:80]

        # Vertical line should have higher values than horizontal after rotation
        assert np.mean(vertical_line_values) > np.mean(horizontal_line_values)