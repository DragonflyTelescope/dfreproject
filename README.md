# dfreproject

A high-performance Python package for reprojecting astronomical images between different coordinate systems with support for SIP distortion correction.

[![Documentation Status](https://readthedocs.org/projects/dfreproject/badge/?version=latest)](https://dfreproject.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/DragonflyTelescope/dfreproject/actions/workflows/tests.yml/badge.svg)](https://github.com/DragonflyTelescope/dfreproject/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/DragonflyTelescope/dfreproject/graph/badge.svg?token=409E407TN5)](https://codecov.io/gh/DragonflyTelescope/dfreproject)


[![DOI](https://zenodo.org/badge/936088731.svg)](https://doi.org/10.5281/zenodo.15170605)

[![DOI](https://joss.theoj.org/papers/10.21105/joss.08525/status.svg)](https://doi.org/10.21105/joss.08525)

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/DragonflyTelescope/dfreproject/badge)](https://scorecard.dev/viewer/?uri=github.com/DragonflyTelescope/dfreproject)

[![License](https://img.shields.io/github/license/DragonflyTelescope/dfreproject)](https://github.com/DragonflyTelescope/dfreproject/blob/main/LICENSE)

The idea behind this package was to make a stripped down version of the `reproject` package affiliated with astropy in order to reduce computational time.
We achieve approximately 20X faster computations with this package using the GPU and 10X using the CPU for images taken by the Dragonfly Telephoto Array. Take a look at the demos for an example.
We note that the only projection we currently support is the Tangential Gnomonic projection which is the most popular in optical astronomy.
If you have need for this package to work with another projection geometry, please open a GitHub issue.

## Features

- Fast reprojection of astronomical images between different WCS frames
- Flux conservation is the default
- Support for Simple Imaging Polynomial (SIP) distortion correction
- GPU acceleration using PyTorch
- Multiple interpolation methods (nearest neighbor, bilinear, bicubic)
- Simple high-level API and detailed low-level control

## Installation

### Using PyPi

If you want to install using PyPi (which is certainly the easiest way), you can simply run

```bash
pip install dfreproject
```

### Requirements

- Python 3.7+
- NumPy
- Astropy
- PyTorch
- Matplotlib
- cmcrameri

### Installing from Source

For the latest development version, install directly from the GitHub repository:

```bash
git clone https://github.com/DragonflyTelescope/dfreproject.git
cd dfreproject
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
from dfreproject import calculate_reprojection

# Load source and target images
source_hdu = fits.open('source_image.fits')[0]
target_hdu = fits.open('target_grid.fits')[0]
target_wcs = WCS(target_hdu.header)
# Perform dfreproject with bilinear interpolation
reprojected = calculate_reprojection(
    source_hdus=source_hdu,
    target_wcs=target_wcs,
    shape_out=target_hdu.data.shape,
    order='bilinear'
)

# Save as FITS
output_hdu = fits.PrimaryHDU(data=reprojected)
output_hdu.header.update(target_wcs.to_header())
output_hdu.writeto('reprojected_image.fits', overwrite=True)
```

The arguments for `calculate_reprojection` are the same as for the standard reprojection options in the reproject package, i.e. `reproject_interp`, `reproject_adaptive`, or `reproject_exact`.

In another scenario, it may be more advantageous to pass a tuple of a data array and a header object that have already been loaded into memory (i.e. not an HDU object). In that case, follow this example:

```python
from astropy.io import fits
from astropy.wcs import WCS
from dfreproject import calculate_reprojection

# Load source and target images
source_hdu = fits.open('source_image.fits')[0]
source_data = source_hdu.data
target_hdu = fits.open('target_grid.fits')[0]
target_wcs = WCS(target_hdu.header)
# Perform dfreproject with bilinear interpolation
reprojected = calculate_reprojection(
    source_hdus=(source_data, source_hdu.header),
    target_wcs=target_wcs,
    shape_out=target_hdu.data.shape,
    order='bilinear'
)

# Save as FITS
output_hdu = fits.PrimaryHDU(data=reprojected)
output_hdu.header.update(target_wcs.to_header())
output_hdu.writeto('reprojected_image.fits', overwrite=True)
```

The `calculate_reprojection` function will internally handle all the translation so that the inputs are properly handled.

### Flux Conservation
Flux conservation is the default behavior for `dfreproject`. Two options are available for flux conservation:
1. Local flux density conservation: The image and a "ones" tensor are interpolated together, and the interpolated image is divided by the interpolated ones tensor (footprint) to correct for any flux density spreading during interpolation. This can affect the results at the edges of the interpolation.
2. Jacobian correction for full flux conservation: Multiply the footprint-corrected flux by the determinant of the Jacobian to handle changes in area during the reprojection.

Local flux conservation is the default option; however, users can change this behavior by setting `conserve_flux=False`. If the transformation between one coordinate system and another is truly linear (i.e., there are no distortions such as SIP distortions), then the local flux convervation computed with the footprint is sufficient. If this is the case, then the user can set `compute_jacobian=False`. However, this only achieves very modest gains in computation time so we suggest users leave this feature on.

## Demos and Examples

A collection of example notebooks and scripts is available in the `demos` folder to help you get started:

- `reprojection-comparison.ipynb` - Simple example of reprojecting between two WCS frames and comparing the result of our implementation with the `reproject` package.
- `reprojection-comparison-mini.ipynb` - Example demonstrating the differences between `dfreproject` and `reproject` using different interpolation schema.
- `Coordinate-Comparison.ipynb` - A step-by-step walkthrough of our coordinate transformations with a comparison to `astropy.wcs`.

To run the demos:

```bash
cd demos
jupyter notebook
```

## Documentation

Comprehensive documentation is available at [https://dfreproject.readthedocs.io/](https://dfreproject.readthedocs.io/)
The documentation includes:

- API reference
- Mathematical details of the reprojection process
- Tutorials and examples
- Performance tips

## Running Tests
The unit tests can be run using the following command:

```bash
pytest
```

The default settings are in the `pytest.ini` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this package in your research, please cite our zenodo DOI:

https://doi.org/10.5281/zenodo.15170605

## Acknowledgments

- Based on the FITS WCS standard and SIP convention
- Inspired by Astropy's reproject package
- Accelerated with PyTorch

The License for all past and present versions is the GPL-3.0.

## AI Disclaimer
`Claude.ai` was used to improve the documentation and help write the unit tests. The only portion of the code that `Claude.ai` wrote is related to the chunking of data to respect memory requirements.
