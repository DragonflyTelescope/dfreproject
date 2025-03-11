import pytest
import torch
import numpy as np

# Import the functions to test - adjust imports as needed
from dfreproject.sip import (
    apply_sip_distortion,
    apply_inverse_sip_distortion,
    iterative_inverse_sip
)


@pytest.mark.unit
class TestSIPFunctions:
    """Tests for SIP distortion correction functions."""

    @pytest.fixture
    def mock_sip_coeffs(self):
        """Create mock SIP coefficients without relying on Astropy."""
        # Create mock SIP coefficients directly
        sip_coeffs = {
            'a_order': 2,
            'b_order': 2,
            'ap_order': 2,
            'bp_order': 2,
            'a': np.zeros((3, 3)),
            'b': np.zeros((3, 3)),
            'ap': np.zeros((3, 3)),
            'bp': np.zeros((3, 3))
        }

        # Set some coefficient values
        sip_coeffs['a'][0, 1] = 0.001  # u*v^0 term
        sip_coeffs['a'][1, 0] = 0.002  # u^0*v term
        sip_coeffs['a'][2, 0] = 0.0001
        sip_coeffs['a'][1, 1] = 0.0002
        sip_coeffs['a'][0, 2] = 0.0003

        sip_coeffs['b'][0, 1] = 0.002
        sip_coeffs['b'][1, 0] = 0.001
        sip_coeffs['b'][2, 0] = 0.0003
        sip_coeffs['b'][1, 1] = 0.0001
        sip_coeffs['b'][0, 2] = 0.0002

        # Set inverse coefficients
        sip_coeffs['ap'][0, 1] = -0.001
        sip_coeffs['ap'][1, 0] = -0.002
        sip_coeffs['ap'][2, 0] = -0.0001
        sip_coeffs['ap'][1, 1] = -0.0002
        sip_coeffs['ap'][0, 2] = -0.0003

        sip_coeffs['bp'][0, 1] = -0.002
        sip_coeffs['bp'][1, 0] = -0.001
        sip_coeffs['bp'][2, 0] = -0.0003
        sip_coeffs['bp'][1, 1] = -0.0001
        sip_coeffs['bp'][0, 2] = -0.0002

        return sip_coeffs

    @pytest.fixture
    def mock_sip_coeffs_no_inverse(self):
        """Create mock SIP coefficients without inverse coefficients."""
        # Create mock SIP coefficients directly
        sip_coeffs = {
            'a_order': 2,
            'b_order': 2,
            'ap_order': 0,  # No inverse
            'bp_order': 0,  # No inverse
            'a': np.zeros((3, 3)),
            'b': np.zeros((3, 3))
        }

        # Set some coefficient values
        sip_coeffs['a'][0, 1] = 0.001
        sip_coeffs['a'][1, 0] = 0.002
        sip_coeffs['a'][2, 0] = 0.0001
        sip_coeffs['a'][1, 1] = 0.0002
        sip_coeffs['a'][0, 2] = 0.0003

        sip_coeffs['b'][0, 1] = 0.002
        sip_coeffs['b'][1, 0] = 0.001
        sip_coeffs['b'][2, 0] = 0.0003
        sip_coeffs['b'][1, 1] = 0.0001
        sip_coeffs['b'][0, 2] = 0.0002

        return sip_coeffs

    def test_apply_sip_distortion(self, mock_sip_coeffs, device):
        """Test applying SIP distortion to coordinates."""
        # Test point away from reference pixel
        u = torch.tensor(10.0, device=device)
        v = torch.tensor(20.0, device=device)

        # Apply distortion
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs, device)

        # Calculate expected results manually
        expected_u_dist = u + (0.001 * v + 0.002 * u + 0.0001 * u * u + 0.0002 * u * v + 0.0003 * v * v)
        expected_v_dist = v + (0.002 * v + 0.001 * u + 0.0003 * u * u + 0.0001 * u * v + 0.0002 * v * v)

        # Check that results match expected values
        assert u_dist.item() == pytest.approx(expected_u_dist.item(), abs=1e-5)
        assert v_dist.item() == pytest.approx(expected_v_dist.item(), abs=1e-5)

    def test_apply_sip_distortion_batch(self, mock_sip_coeffs, device):
        """Test applying SIP distortion to batches of coordinates."""
        # Create a batch of coordinates
        u = torch.tensor([10.0, 20.0, 30.0], device=device)
        v = torch.tensor([20.0, 30.0, 40.0], device=device)

        # Apply distortion to batch
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs, device)

        # Check batch size is maintained
        assert u_dist.shape == u.shape
        assert v_dist.shape == v.shape

        # Check each coordinate was properly distorted
        for i in range(len(u)):
            u_i = u[i]
            v_i = v[i]
            expected_u_dist = u_i + (
                        0.001 * v_i + 0.002 * u_i + 0.0001 * u_i * u_i + 0.0002 * u_i * v_i + 0.0003 * v_i * v_i)
            expected_v_dist = v_i + (
                        0.002 * v_i + 0.001 * u_i + 0.0003 * u_i * u_i + 0.0001 * u_i * v_i + 0.0002 * v_i * v_i)

            assert u_dist[i].item() == pytest.approx(expected_u_dist.item(), abs=1e-5)
            assert v_dist[i].item() == pytest.approx(expected_v_dist.item(), abs=1e-5)

    def test_apply_sip_distortion_2d_grid(self, mock_sip_coeffs, device):
        """Test applying SIP distortion to a 2D grid."""
        # Create a 2D grid
        u, v = torch.meshgrid(
            torch.linspace(0, 50, 5, device=device),
            torch.linspace(0, 50, 5, device=device),
            indexing='ij'
        )

        # Apply distortion
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs, device)

        # Check shape is maintained
        assert u_dist.shape == u.shape
        assert v_dist.shape == v.shape

        # Check a few sample points
        i, j = 2, 3  # Some indices in the grid
        u_ij = u[i, j]
        v_ij = v[i, j]
        expected_u_dist = u_ij + (
                    0.001 * v_ij + 0.002 * u_ij + 0.0001 * u_ij * u_ij + 0.0002 * u_ij * v_ij + 0.0003 * v_ij * v_ij)
        expected_v_dist = v_ij + (
                    0.002 * v_ij + 0.001 * u_ij + 0.0003 * u_ij * u_ij + 0.0001 * u_ij * v_ij + 0.0002 * v_ij * v_ij)

        assert u_dist[i, j].item() == pytest.approx(expected_u_dist.item(), abs=1e-5)
        assert v_dist[i, j].item() == pytest.approx(expected_v_dist.item(), abs=1e-5)

    def test_apply_inverse_sip_distortion(self, mock_sip_coeffs, device):
        """Test applying inverse SIP distortion to coordinates."""
        # Test point
        u = torch.tensor(10.0, device=device)
        v = torch.tensor(20.0, device=device)

        # First apply distortion to get distorted coordinates
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs, device)

        # Now apply inverse distortion
        u_corr, v_corr = apply_inverse_sip_distortion(u_dist, v_dist, mock_sip_coeffs, device)

        # Should get back close to the original coordinates
        assert u_corr.item() == pytest.approx(u.item(), abs=1e-2)
        assert v_corr.item() == pytest.approx(v.item(), abs=1e-2)

    def test_inverse_sip_with_no_inverse_coeffs(self, mock_sip_coeffs_no_inverse, device):
        """Test applying inverse SIP when no inverse coefficients are available."""
        # Test point
        u = torch.tensor(10.0, device=device)
        v = torch.tensor(20.0, device=device)

        # Apply distortion
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs_no_inverse, device)

        # Apply inverse distortion - should use iterative method
        u_corr, v_corr = apply_inverse_sip_distortion(u_dist, v_dist, mock_sip_coeffs_no_inverse, device)

        # Should still recover original coordinates with good precision
        assert u_corr.item() == pytest.approx(u.item(), abs=1e-4)
        assert v_corr.item() == pytest.approx(v.item(), abs=1e-4)

    def test_iterative_inverse_sip(self, mock_sip_coeffs, device):
        """Test the iterative inverse SIP algorithm directly."""
        # Test point
        u = torch.tensor(10.0, device=device)
        v = torch.tensor(20.0, device=device)

        # First apply distortion
        u_dist, v_dist = apply_sip_distortion(u, v, mock_sip_coeffs, device)

        # Now apply iterative inverse
        u_corr, v_corr = iterative_inverse_sip(
            u_dist, v_dist, mock_sip_coeffs, device, max_iter=10, tol=1e-8
        )

        # Should get back close to the original coordinates
        assert u_corr.item() == pytest.approx(u.item(), abs=1e-5)
        assert v_corr.item() == pytest.approx(v.item(), abs=1e-5)

    def test_sip_null_case(self, device):
        """Test that SIP functions return input when no distortion is provided."""
        # Direct return when sip_coeffs is None
        u = torch.tensor(10.0, device=device)
        v = torch.tensor(20.0, device=device)

        # Functions should return input unchanged when sip_coeffs is None
        u_out, v_out = apply_sip_distortion(u, v, None, device)
        assert u_out is u
        assert v_out is v

        u_out, v_out = apply_inverse_sip_distortion(u, v, None, device)
        assert u_out is u
        assert v_out is v