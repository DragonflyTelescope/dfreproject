import pytest
import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os

# Import the function to test
from dfreproject.reproject import calculate_reprojection, Reproject

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output


@pytest.mark.integration
class TestCalculateReprojection:
    """Tests for the high-level calculate_reprojection function."""

    @pytest.fixture
    def simple_source_target_pair(self, test_data_dir):
        """Create a simple source and target HDU pair for testing."""
        # Create a source HDU with a simple pattern
        source_wcs = WCS(naxis=2)
        source_wcs.wcs.crpix = [50, 50]
        source_wcs.wcs.cdelt = [0.1, 0.1]
        source_wcs.wcs.crval = [0, 0]
        source_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create source data - Gaussian blob in the center
        source_data = np.zeros((100, 100), dtype=np.float32)
        y, x = np.mgrid[0:100, 0:100]
        source_data = 100.0 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / 100.0)

        # Create source HDU
        source_hdu = fits.PrimaryHDU(data=source_data)
        source_hdu.header.update(source_wcs.to_header())

        # DEBUG: Print source WCS information
        print("\nSource WCS information:")
        print(f"CRPIX: {source_wcs.wcs.crpix}")
        print(f"CDELT: {source_wcs.wcs.cdelt}")
        print(f"CRVAL: {source_wcs.wcs.crval}")
        print(f"CTYPE: {source_wcs.wcs.ctype}")
        print(f"PC Matrix: {source_wcs.wcs.get_pc()}")

        # Create target HDU with a slightly different WCS
        target_wcs = WCS(naxis=2)
        target_wcs.wcs.crpix = [51, 51]  # Shifted reference pixel
        target_wcs.wcs.cdelt = [0.1, 0.1]
        target_wcs.wcs.crval = [0, 0]
        target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # DEBUG: Print target WCS information
        print("\nTarget WCS information:")
        print(f"CRPIX: {target_wcs.wcs.crpix}")
        print(f"CDELT: {target_wcs.wcs.cdelt}")
        print(f"CRVAL: {target_wcs.wcs.crval}")
        print(f"CTYPE: {target_wcs.wcs.ctype}")
        print(f"PC Matrix: {target_wcs.wcs.get_pc()}")

        # Create target data - empty array
        target_data = np.zeros((120, 120), dtype=np.float32)  # Larger dimensions

        # Create target HDU
        target_hdu = fits.PrimaryHDU(data=target_data)
        target_hdu.header.update(target_wcs.to_header())

        # Save HDUs to files for testing
        source_file = os.path.join(test_data_dir, "simple_source.fits")
        target_file = os.path.join(test_data_dir, "simple_target.fits")

        source_hdu.writeto(source_file, overwrite=True)
        target_hdu.writeto(target_file, overwrite=True)

        return source_hdu, target_hdu, source_file, target_file, target_wcs

    def test_basic_reprojection(self, simple_source_target_pair):
        """Test that the function performs basic dfreproject correctly."""
        source_hdu, target_hdu, _, _, target_wcs = simple_source_target_pair


        # Perform dfreproject
        result = calculate_reprojection(
            source_hdus=source_hdu,
            target_wcs=target_wcs,
            shape_out=target_hdu.data.shape,
            interpolation_mode="bilinear"
        )



        # The result should have a gaussian peak somewhere
        assert not np.isnan(result).all()


    def test_all_interpolation_modes(self, simple_source_target_pair):
        """Test all supported interpolation modes."""
        source_hdu, target_hdu, _, _, target_wcs = simple_source_target_pair

        modes = ["nearest", "bilinear", "bicubic"]

        for mode in modes:
            # Perform dfreproject with each mode
            result = calculate_reprojection(
                source_hdus=source_hdu,
                target_wcs=target_wcs,
                shape_out=target_hdu.data.shape,
                interpolation_mode=mode
            )

            # Check basic properties

            assert result.shape == target_hdu.data.shape
            assert not np.isnan(result).all()


    def test_with_real_fits_files(self, source_fits_file, target_fits_file):
        """Test dfreproject with real FITS files from fixtures."""
        # Load HDUs
        source_hdu = fits.open(source_fits_file)[0]
        target_hdu = fits.open(target_fits_file)[0]

        # Convert data to native byte order if needed
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        if hasattr(target_hdu, 'data') and target_hdu.data is not None:
            target_hdu.data = np.asarray(target_hdu.data, dtype=np.float64).copy()

        # Perform dfreproject
        result = calculate_reprojection(
            source_hdus=source_hdu,
            target_wcs=WCS(target_hdu.header),
            shape_out=target_hdu.data.shape,
            interpolation_mode="bilinear"
        )

        # Check result
        assert result.shape == target_hdu.data.shape
        assert not np.isnan(result).all()

    def test_end_to_end_workflow(self, simple_source_target_pair, test_data_dir):
        """Test a complete workflow including saving results to a new FITS file."""
        _, _, source_file, target_file, target_wcs = simple_source_target_pair

        # Load from files
        source_hdu = fits.open(source_file)[0]
        target_hdu = fits.open(target_file)[0]

        # Convert data to native byte order if needed
        if hasattr(source_hdu, 'data') and source_hdu.data is not None:
            source_hdu.data = np.asarray(source_hdu.data, dtype=np.float64).copy()

        if hasattr(target_hdu, 'data') and target_hdu.data is not None:
            target_hdu.data = np.asarray(target_hdu.data, dtype=np.float64).copy()

        # Perform dfreproject
        reprojected = calculate_reprojection(
            source_hdus=source_hdu,
            target_wcs=WCS(target_hdu.header),
            shape_out=target_hdu.data.shape,
            interpolation_mode="bilinear"
        )

        # Convert to numpy and save as FITS
        reprojected_np = reprojected
        output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
        output_file = os.path.join(test_data_dir, "reprojected_output.fits")
        output_hdu.writeto(output_file, overwrite=True)

        # Verify the saved file exists and can be read
        assert os.path.exists(output_file)
        verification_hdu = fits.open(output_file)[0]
        assert verification_hdu.data.shape == target_hdu.data.shape



    def test_correct_reproject_instance_created(self, simple_source_target_pair, monkeypatch):
        """Test that the function creates a Reproject instance with the correct parameters."""
        source_hdu, target_hdu, _, _, target_wcs = simple_source_target_pair

        # Create a mock to track how Reproject is called
        original_init = Reproject.__init__
        calls = []

        def mock_init(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original_init(self, *args, **kwargs)

        # Apply the mock
        monkeypatch.setattr(Reproject, '__init__', mock_init)

        # Mock the interpolate_source_image method to return a dummy tensor
        original_interpolate = Reproject.interpolate_source_image

        def mock_interpolate(self, interpolation_mode="nearest"):
            # Return a dummy tensor with the right shape
            return torch.zeros(target_hdu.data.shape)

        monkeypatch.setattr(Reproject, 'interpolate_source_image', mock_interpolate)

        # Call the function
        result = calculate_reprojection(
            source_hdus=source_hdu,
            target_wcs=target_wcs,
            shape_out=target_hdu.data.shape,
            interpolation_mode="bilinear"
        )

        # Check that Reproject was initialized correctly
        assert len(calls) == 1
        init_args, init_kwargs = calls[0]

        # Check source_hdus and target_hdu were passed correctly
        assert source_hdu in init_kwargs.get('source_hdus')
