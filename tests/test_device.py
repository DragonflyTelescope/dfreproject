import pytest
import torch
import unittest.mock as mock

# Import the function to test
from dfreproject.utils import get_device  # Adjust import path as needed


@pytest.mark.unit
class TestDeviceDetection:
    """Tests for the device detection function."""

    def test_get_device_returns_device_object(self):
        """Test that get_device returns a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cuda_available(self):
        """Test that get_device returns CUDA device when available."""
        # Mock torch.cuda to simulate CUDA being available
        with mock.patch('torch.cuda.is_available', return_value=True), \
                mock.patch('torch.cuda.device_count', return_value=1):
            device = get_device()
            assert device.type == 'cuda'
            assert device.index == 0

    def test_get_device_cuda_not_available(self):
        """Test that get_device returns CPU when CUDA is not available."""
        # Mock torch.cuda to simulate CUDA not being available
        with mock.patch('torch.cuda.is_available', return_value=False):
            device = get_device()
            assert device.type == 'cpu'

    def test_get_device_no_cuda_devices(self):
        """Test that get_device returns CPU when no CUDA devices are found."""
        # Mock torch.cuda to simulate CUDA being available but no devices
        with mock.patch('torch.cuda.is_available', return_value=True), \
                mock.patch('torch.cuda.device_count', return_value=0):
            device = get_device()
            assert device.type == 'cpu'

    def test_get_device_handles_exceptions(self):
        """Test that get_device handles exceptions and falls back to CPU."""
        # Mock torch.cuda.is_available to raise an exception
        with mock.patch('torch.cuda.is_available', side_effect=RuntimeError("CUDA error")):
            # Capture stdout to verify the error message
            with mock.patch('builtins.print') as mock_print:
                device = get_device()
                # Check that it fell back to CPU
                assert device.type == 'cpu'
                # Check that error message was printed
                assert mock_print.call_count >= 1
                assert any("CUDA error" in str(args) for args, _ in mock_print.call_args_list)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_device_real_cuda(self):
        """Test get_device with real CUDA if available (integration test)."""
        device = get_device()
        # If the system has CUDA, this should return a CUDA device
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            assert device.type == 'cuda'
            assert device.index == 0
