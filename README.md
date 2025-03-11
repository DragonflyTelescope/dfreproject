# Astronomical Image Reprojection

A high-performance Python package for reprojecting astronomical images between different coordinate systems with support for SIP distortion correction.

[![Documentation Status](https://readthedocs.org/projects/reprojection/badge/?version=latest)](https://reprojection.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/DragonflyTelescope/reprojection/actions/workflows/tests.yml/badge.svg)](https://github.com/DragonflyTelescope/reprojection/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/DragonflyTelescope/reprojection/branch/main/graph/badge.svg)](https://codecov.io/gh/DragonflyTelescope/reprojection)

The idea behind this package was to make a stripped down version of the `reproject` package by astropy in order to reduce computational time.
We achieve approximately 40X faster computations with this package. Take a look at the demos for an example.

## Features

- Fast reprojection of astronomical images between different WCS frames
- Support for Simple Imaging Polynomial (SIP) distortion correction
- GPU acceleration using PyTorch
- Multiple interpolation methods (nearest neighbor, bilinear, bicubic)
- Simple high-level API and detailed low-level control

## Installation

### Requirements

- Python 3.7+
- NumPy
- Astropy
- PyTorch
- Matplotlib (for visualization)
- cmcrameri

### Installing from Source

For the latest development version, install directly from the GitHub repository:

```bash
git clone https://github.com/dragonfly/reprojection.git
cd reprojection
pip install -e .
```

For development installation with documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Quick Start

```python
from astropy.io import fits
from astropy.wcs import WCS
from reprojection import calculate_reprojection

# Load source and target images
source_hdu = fits.open('source_image.fits')[0]
target_hdu = fits.open('target_grid.fits')[0]
target_wcs = WCS(target_hdu.header)
# Perform reprojection with bilinear interpolation
reprojected = calculate_reprojection(
    source_hdus=source_hdu,
    target_wcs=target_wcs,
    shape_out=target_hdu.data.shape,
    interpolation_mode='bilinear'
)

# Convert back to NumPy and save as FITS
reprojected_np = reprojected.cpu().numpy()
output_hdu = fits.PrimaryHDU(data=reprojected_np)
output_hdu.header.update(target_wcs.to_header())
output_hdu.writeto('reprojected_image.fits', overwrite=True)
```

## Demos and Examples

A collection of example notebooks and scripts is available in the `demos` folder to help you get started:

- `reprojection-comparison.ipynb` - Simple example of reprojecting between two WCS frames and comparing the result of our implementation with the `reproject` package.
- `Coordinate-Comparison.ipynb' - A step-by-step walkthrough of our coordinate transformations with a comparison to `astropy.wcs`.

To run the demos:

```bash
cd demos
jupyter notebook
```

## Documentation

Comprehensive documentation is available at [https://reprojection.readthedocs.io/](https://reprojection.readthedocs.io/)

The documentation includes:
- API reference
- Mathematical details of the reprojection process
- Tutorials and examples
- Performance tips

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:



## Acknowledgments

- Based on the FITS WCS standard and SIP convention
- Inspired by Astropy's reproject package
- Accelerated with PyTorch
- Documentation aided by Claude.ai
