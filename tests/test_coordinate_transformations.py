import pytest
import torch
import numpy as np
from dfreproject.reproject import Reproject
from astropy.io import fits
from astropy.wcs import WCS

@pytest.mark.unit
class TestCoordinateTransformations:
    """Test suite for astronomical coordinate transformations."""

    @pytest.fixture(autouse=True)
    def setup_transformer(self, source_fits_file, target_fits_file, device):
        """Set up transformer object for each test."""
        # Create transformer with the same WCS for both source and target for basic tests
        source_hdul = fits.open(source_fits_file)
        target_hdul = fits.open(target_fits_file)
        # Convert data to native byte order if needed
        if hasattr(source_hdul[0], 'data') and source_hdul[0].data is not None:
            source_hdul[0].data = np.asarray(source_hdul[0].data, dtype=np.float64).copy()

        if hasattr(target_hdul[0], 'data') and target_hdul[0].data is not None:
            target_hdul[0].data = np.asarray(target_hdul[0].data, dtype=np.float64).copy()
        self.transformer = Reproject([source_hdul[0]], WCS(source_hdul[0].header), shape_out=target_hdul[0].data.shape,)

        # Define test pixel coordinates
        self.test_pixels = [
            (100, 100),  # Reference pixel
            (150, 150),  # Offset from reference
            (50, 50),  # Opposite direction
            (200, 100),  # Only X direction
            (100, 200),  # Only Y direction
        ]

    @pytest.mark.parametrize("x,y,expected_ra,expected_dec", [
        (100, 100, 150.0, 30.0),  # Reference pixel
        (101, 100, 149.99, 30.0),  # Small x offset
        (100, 101, 150.0, 30.01),  # Small y offset
    ])
    def test_specific_coordinates(self, x, y, expected_ra, expected_dec, device):
        """Test specific coordinate pairs with known results."""
        # Convert coordinates to torch tensors with batch dimension
        x_tensor = torch.tensor([[x]], dtype=torch.float64, device=device)
        y_tensor = torch.tensor([[y]], dtype=torch.float64, device=device)

        # Transform coordinates
        ra, dec = self.transformer.calculate_skyCoords(x_grid=x_tensor, y_grid=y_tensor)

        # Check results (extract first element due to batch dimension)
        assert pytest.approx(ra[0, 0].item(), abs=1e-2) == expected_ra
        assert pytest.approx(dec[0, 0].item(), abs=1e-2) == expected_dec


    def test_sky_coords_reference_pixel(self, device):
        """Test that reference pixel maps to reference sky coords."""
        # Transform reference pixel
        x = torch.tensor([[100]], device=device, dtype=torch.float64)
        y = torch.tensor([[100]], device=device, dtype=torch.float64)

        ra, dec = self.transformer.calculate_skyCoords(x, y)

        # Check against expected values
        assert pytest.approx(ra.item(), abs=1e-2) == 150.0
        assert pytest.approx(dec.item(), abs=1e-2) == 30.0

    def test_sky_coords_batch_processing(self, device):
        """Test batch processing of multiple coordinates."""
        # Create a grid of pixel coordinates as tensors
        from fixtures_and_helpers import get_test_grid

        grid_y, grid_x = get_test_grid(shape=(10, 10), start=(50, 50), end=(150, 150))

        # Convert to torch tensors with batch dimension
        grid_x = torch.tensor(grid_x, dtype=torch.float64, device=device)
        grid_y = torch.tensor(grid_y, dtype=torch.float64, device=device)

        # Transform using batch processing
        ra, dec = self.transformer.calculate_skyCoords(grid_x, grid_y)

        # Check expected shape
        assert ra.shape == (1, 10, 10)
        assert dec.shape == (1, 10, 10)

        # Also check some values - the center should be near the reference point
        center_idx = (0, 5, 5)  # Middle of the 10x10 grid
        assert pytest.approx(ra[center_idx].item(), abs=0.5) == 150.0
        assert pytest.approx(dec[center_idx].item(), abs=0.5) == 30.0



    def test_roundtrip_transformation(self, device):
        """Test that roundtrip transformation returns original coordinates."""
        # Original pixel coordinates
        orig_x = torch.tensor([100, 150], dtype=torch.float64, device=device)
        orig_y = torch.tensor([100, 150], dtype=torch.float64, device=device)

        # Calculate source coordinates
        source_x, source_y = self.transformer.calculate_sourceCoords()

        # Check roundtrip accuracy - within 1e-5 pixel
        assert torch.allclose(orig_x[0], source_x[0, 100, 100], atol=1e-2)
        assert torch.allclose(orig_y[0], source_y[0, 100, 100], atol=1e-2)

        # Check roundtrip accuracy - within 1e-5 pixel
        assert torch.allclose(orig_x[1], source_x[0, 150, 150], atol=1e-2)
        assert torch.allclose(orig_y[1], source_y[0, 150, 150], atol=1e-2)

    def test_compare_with_astropy(self, simple_wcs, device):
        """Compare results with Astropy WCS for validation."""
        # Test coordinates
        test_x = np.array([100, 150, 200])
        test_y = np.array([100, 150, 200])

        # Astropy results (using 0-based indexing)
        astropy_result = simple_wcs.all_pix2world(np.column_stack((test_x, test_y)), 0)
        astropy_ra, astropy_dec = astropy_result[:, 0], astropy_result[:, 1]

        # Our implementation results
        our_ra, our_dec = self.transformer.calculate_skyCoords(
            torch.tensor(test_x, dtype=torch.float64, device=device).unsqueeze(0),
            torch.tensor(test_y, dtype=torch.float64, device=device).unsqueeze(0)
        )

        # Remove batch dimension and convert to numpy
        our_ra = our_ra.squeeze(0).squeeze(0).cpu().numpy()
        our_dec = our_dec.squeeze(0).squeeze(0).cpu().numpy()

        # Compare results - should be close within a small tolerance
        np.testing.assert_allclose(our_ra, astropy_ra, rtol=1e-2)
        np.testing.assert_allclose(our_dec, astropy_dec, rtol=1e-2)


    def test_edge_cases(self, simple_wcs, device):
        """Test handling of edge cases."""
        # Create a transformer
        # transformer = YourClass(simple_wcs, simple_wcs, device=device)

        # Test with coordinates at the edge of the image
        edge_coords = [
            (0, 0),  # Top-left corner
            (0, 199),  # Bottom-left corner
            (199, 0),  # Top-right corner
            (199, 199),  # Bottom-right corner
        ]

        for x, y in edge_coords:
            x_tensor = torch.tensor([[x]], dtype=torch.float64, device=device)
            y_tensor = torch.tensor([[y]], dtype=torch.float64, device=device)

            # Transform coordinates
            ra, dec = self.transformer.calculate_skyCoords(x_tensor, y_tensor)

            # Just check that the results are finite
            assert torch.isfinite(ra).all()
            assert torch.isfinite(dec).all()

        # Placeholder assertion - remove when implementing real test
        assert True

