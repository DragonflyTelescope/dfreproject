.. AstroReproject documentation master file

Welcome to dfreproject's documentation!
========================================

.. image:: https://codecov.io/gh/DragonflyTelescope/dfreproject/graph/badge.svg?token=409E407TN5
 :target: https://codecov.io/gh/DragonflyTelescope/dfreproject

.. image:: https://zenodo.org/badge/936088731.svg
  :target: https://doi.org/10.5281/zenodo.15170605

.. image:: https://readthedocs.org/projects/dfreproject/badge/?version=latest
    :target: https://dfreproject.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

dfreproject is a Python package for reprojecting astronomical images. The code runs using torch in order to speed up calculations.

The idea behind this package was to make a stripped down version of the `reproject` package affiliated with astropy in order to reduce computational time.
We achieve approximately 20X faster computations with this package using the GPU and 10X using the CPU for images taken by the Dragonfly Telephoto Array. Take a look at the demos for an example.
We note that the only projection we currently support is the Tangential Gnomonic projection which is the most popular in optical astronomy.
If you have need for this package to work with another projection geometry, please open a GitHub ticket.

For the sake of transparency of the calculations `dfreproject` performs, we've written out the computations at each step in the `methodology section <methodology.rst>`_ file.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   methodology
   examples


You can quickly run `dfreproject` with the following:


.. code-block:: python

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



If you use this package, please cite our Zenodo DOI:

https://doi.org/10.5281/zenodo.15170605



Note: Claude.ai was used in preparing the docstrings for the functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
