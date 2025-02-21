Astronomical Image Reprojection
==============================

Mathematical Foundation
---------------------

Astronomical image reprojection is the process of transforming an image from one World Coordinate System (WCS) to another. This involves converting pixel positions from one coordinate frame to another while preserving the underlying celestial coordinates. This document explains the mathematical principles and implementation details of our reprojection process.

The Complete Transformation Pipeline
-----------------------------------

The reprojection process follows these steps:

1. Calculate sky coordinates (RA/Dec) for each pixel in the target image
2. Convert these sky coordinates to pixel coordinates in the source image
3. Interpolate the source image at these calculated coordinates

Each of these steps involves several mathematical transformations, which we'll detail below.

WCS Transformation with SIP Distortion
-------------------------------------

Our implementation handles the FITS World Coordinate System (WCS) standard with Simple Imaging Polynomial (SIP) distortion correction. WCS with SIP defines a mapping between pixel coordinates and celestial coordinates.

Pixel to Sky Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

To convert from pixel coordinates (x, y) to sky coordinates (RA, Dec), we follow this process:

1. **Pixel Offset Calculation**

   We first compute the offset from the reference pixel:

   .. math::

      u = x - \text{CRPIX1} \\
      v = y - \text{CRPIX2}

   where CRPIX1 and CRPIX2 are the reference pixel coordinates.

2. **SIP Distortion Correction**

   We apply the SIP distortion polynomials to account for optical distortion:

   .. math::

      x' = u + A(u, v) \\
      y' = v + B(u, v)

   The SIP polynomials A and B are defined as:

   For order 2:

   .. math::

      A(u, v) = A_{0,2} v^2 + A_{1,1} u v + A_{2,0} u^2 \\
      B(u, v) = B_{0,2} v^2 + B_{1,1} u v + B_{2,0} u^2

   For order 3:

   .. math::

      A(u, v) &= A_{0,2} v^2 + A_{0,3} v^3 + A_{1,1} u v + A_{1,2} u v^2 + A_{2,0} u^2 + A_{2,1} u^2 v + A_{3,0} u^3 \\
      B(u, v) &= B_{0,2} v^2 + B_{0,3} v^3 + B_{1,1} u v + B_{1,2} u v^2 + B_{2,0} u^2 + B_{2,1} u^2 v + B_{3,0} u^3

3. **PC Matrix Transformation**

   The distorted coordinates are transformed by the PC (Projection Coordinate) matrix:

   .. math::

      \begin{pmatrix} x'' \\ y'' \end{pmatrix} = 
      \begin{pmatrix} \text{PC1\_1} & \text{PC1\_2} \\ \text{PC2\_1} & \text{PC2\_2} \end{pmatrix}
      \begin{pmatrix} x' \\ y' \end{pmatrix}

4. **Scaling by CDELT**

   The coordinates are scaled by the pixel scale factors:

   .. math::

      x''' = x'' \cdot \text{CDELT1} \\
      y''' = y'' \cdot \text{CDELT2}

5. **Projection onto the Celestial Sphere**

   The intermediate coordinates are projected onto the celestial sphere using the gnomonic (tangent plane) projection:

   .. math::

      r &= \sqrt{(x''')^2 + (y''')^2} \\
      \theta &= \arctan(r) \\
      \phi &= \arctan2(-x''', y''')

   The celestial coordinates are then:

   .. math::

      \delta &= \arcsin(\sin(\delta_0) \cos(\theta) + \cos(\delta_0) \sin(\theta) \cos(\phi)) \\
      \alpha &= \alpha_0 + \arctan2(\sin(\theta) \sin(\phi), \cos(\delta_0) \cos(\theta) - \sin(\delta_0) \sin(\theta) \cos(\phi))

   where :math:`\alpha_0` and :math:`\delta_0` are the reference RA and Dec (CRVAL1 and CRVAL2) in radians.

Sky to Pixel Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~

The inverse transformation from sky coordinates (RA, Dec) to pixel coordinates involves these steps:

1. **Compute Direction Cosines**

   From sky coordinates, we compute the angular difference from the reference point:

   .. math::

      \Delta\alpha &= \alpha - \alpha_0 \\
      \cos(\Delta\alpha) &= \cos(\Delta\alpha) \\
      \sin(\Delta\alpha) &= \sin(\Delta\alpha)

   We ensure that :math:`\Delta\alpha` is in the range :math:`[-\pi, \pi]` to handle the RA wrap-around.

2. **Compute Tangent Plane Coordinates**

   We project onto the tangent plane:

   .. math::

      \text{denom} &= \sin(\delta_0) \sin(\delta) + \cos(\delta_0) \cos(\delta) \cos(\Delta\alpha) \\
      x''' &= -\cos(\delta) \sin(\Delta\alpha) / \text{denom} \\
      y''' &= (\sin(\delta) \cos(\delta_0) - \cos(\delta) \sin(\delta_0) \cos(\Delta\alpha)) / \text{denom}

   A small epsilon (e.g., 1e-10) is added to the denominator to ensure numerical stability.

3. **Scale by Inverse CDELT**

   We scale by the inverse of the pixel scale factors:

   .. math::

      x'' &= x''' / \text{CDELT1} \\
      y'' &= y''' / \text{CDELT2}

4. **Apply Inverse PC Matrix**

   We apply the inverse of the PC matrix:

   .. math::

      \begin{pmatrix} x' \\ y' \end{pmatrix} = 
      \begin{pmatrix} \text{PC1\_1} & \text{PC1\_2} \\ \text{PC2\_1} & \text{PC2\_2} \end{pmatrix}^{-1}
      \begin{pmatrix} x'' \\ y'' \end{pmatrix}

5. **Inverse SIP Distortion Correction**

   We apply the inverse SIP distortion polynomials:

   .. math::

      x_{\text{undistorted}} &= x' + AP(x', y') \\
      y_{\text{undistorted}} &= y' + BP(x', y')

   The inverse SIP polynomials AP and BP have similar forms to A and B but include constant and first-order terms:

   For order 2:

   .. math::

      AP(x', y') &= AP_{0,0} + AP_{0,1} y' + AP_{0,2} (y')^2 + AP_{1,0} x' + AP_{1,1} x' y' + AP_{2,0} (x')^2 \\
      BP(x', y') &= BP_{0,0} + BP_{0,1} y' + BP_{0,2} (y')^2 + BP_{1,0} x' + BP_{1,1} x' y' + BP_{2,0} (x')^2

   For order 3, additional higher-order terms are included.

6. **Add Reference Pixel**

   Finally, we add the reference pixel coordinates to get the final pixel position:

   .. math::

      x &= x_{\text{undistorted}} + \text{CRPIX1} \\
      y &= y_{\text{undistorted}} + \text{CRPIX2}

Interpolation Methods
-------------------

Once we have the mapping from target pixels to source pixels, we need to interpolate the source image values. Our implementation supports three interpolation methods:

1. **Nearest Neighbor**

   The simplest method, which takes the value of the nearest pixel:

   .. math::

      v(x,y) = v(\lfloor x + 0.5 \rfloor, \lfloor y + 0.5 \rfloor)

   This method is fastest but produces blocky results.

2. **Bilinear Interpolation**

   Uses the weighted average of the four nearest pixels:

   .. math::

      v(x,y) = &v(x_0,y_0)(1-dx)(1-dy) + v(x_1,y_0)dx(1-dy) + \\
              &v(x_0,y_1)(1-dx)dy + v(x_1,y_1)dxdy

   Where :math:`dx = x - x_0` and :math:`dy = y - y_0`.

   This provides smoother results than nearest neighbor.

3. **Bicubic Interpolation**

   Uses a 4×4 neighborhood of pixels with cubic weighting:

   .. math::

      v(x,y) = \sum_{i=0}^{3}\sum_{j=0}^{3} a_{ij} x^i y^j

   This provides the highest quality results but is computationally more expensive.

GPU Acceleration
--------------

Our implementation leverages PyTorch for GPU acceleration. The coordinate transformations and interpolation are performed on the GPU if available, providing significant speed improvements for large images.

The key components that benefit from GPU acceleration are:

1. Matrix operations (PC matrix transformation)
2. Polynomial evaluations (SIP distortion calculations)
3. Trigonometric functions (projection calculations)
4. Grid interpolation (using torch.nn.functional.grid_sample)

Implementation Details
--------------------

The reprojection process is implemented in the following classes:

1. **WCSHeader**: Stores all WCS parameters and implements the SIP polynomial functions
2. **Reproject**: Handles the coordinate transformations and interpolation

The high-level API function `calculate_reprojection` provides a convenient interface for users, automatically handling device selection and data type conversions.

Usage Example
-----------

Here's a complete example of reprojecting an image:

.. code-block:: python

    from astropy.io import fits
    from astroreproject import calculate_reprojection
    
    # Load source and target images
    source_hdu = fits.open('source_image.fits')[0]
    target_hdu = fits.open('target_grid.fits')[0]
    
    # Perform reprojection with bilinear interpolation
    reprojected = calculate_reprojection(
        source_hdu=source_hdu,
        target_hdu=target_hdu,
        interpolation_mode='bilinear'
    )
    
    # Convert back to NumPy and save as FITS
    reprojected_np = reprojected.cpu().numpy()
    output_hdu = fits.PrimaryHDU(data=reprojected_np, header=target_hdu.header)
    output_hdu.writeto('reprojected_image.fits', overwrite=True)

Limitations and Future Work
-------------------------

The current implementation has some limitations:

1. Supports only gnomonic (TAN) projection with SIP distortion
2. Handles only 2D images (not data cubes)
3. Does not preserve flux in the strictest sense

Future improvements could include:

1. Support for additional projection types (SIN, CAR, etc.)
2. Flux-conserving resampling methods
3. Uncertainty propagation
4. Support for masked pixels and NaN handling

References
---------

1. Calabretta, M. R., & Greisen, E. W. (2002). Representations of celestial coordinates in FITS. Astronomy & Astrophysics, 395(3), 1077-1122.

2. Shupe, D. L., et al. (2005). The SIP Convention for Representing Distortion in FITS Image Headers. Astronomical Data Analysis Software and Systems XIV, 347, 491.

3. Greisen, E. W., & Calabretta, M. R. (2002). Representations of world coordinates in FITS. Astronomy & Astrophysics, 395(3), 1061-1075.

Note
----
Claude.ai was used in creating this page