Astronomical Coordinate Transformations
=======================================

This module provides PyTorch-based implementations of astronomical coordinate transformations using the World Coordinate System (WCS) standard. These functions allow conversion between pixel coordinates and sky coordinates (RA/Dec) with support for the Tangential (TAN) projection and SIP distortion corrections.

Key WCS Variables
-----------------

- **CRPIX1, CRPIX2**: Reference pixel coordinates (1-based in FITS convention)
- **CRVAL1, CRVAL2**: Reference point sky coordinates (RA, Dec) in degrees
- **PC_matrix**: Transformation matrix that encodes rotation
- **CDELT**: Pixel scale factors (degrees per pixel)
- **CD_matrix**: Combined transformation matrix (PC_matrix * CDELT)
- **SIP coefficients**: Parameters for the Simple Imaging Polynomial distortion correction

calculate_skyCoords
-------------------

.. py:function:: calculate_skyCoords(self, x=None, y=None)

   Convert pixel coordinates in the target image to sky coordinates (RA, Dec).

   This function implements the forward transformation from pixel space to sky coordinates
   following the same logic as the Astropy WCS implementation, but using PyTorch for
   GPU acceleration and batch processing.

   Takes optional pixel coordinates (x, y) as input and returns a tuple of (RA, Dec) coordinates in degrees.

   **Key Steps:**

   1. **Preprocessing**: Compute pixel offsets from the reference pixel (CRPIX)

      .. math::
         u = x - (CRPIX1 - 1)

         v = y - (CRPIX2 - 1)

      where :math:`x, y` are input pixel coordinates and :math:`CRPIX1, CRPIX2` are reference pixel coordinates.

   2. **SIP Correction**: Apply SIP distortion correction if present

      .. math::
         (u', v') = \text{apply_sip_distortion}(u, v)

      where :math:`u', v'` are the distorted coordinates after applying the SIP polynomial.

   3. **Linear Transformation**: Apply the CD matrix (rotation and scaling)

      .. math::
         CD\_matrix = PC\_matrix \cdot CDELT

         \begin{bmatrix} x_{scaled} \\ y_{scaled} \end{bmatrix} = CD\_matrix \cdot \begin{bmatrix} u' \\ v' \end{bmatrix}

      where :math:`PC\_matrix` is the coordinate rotation matrix, :math:`CDELT` contains pixel scale factors,
      and :math:`x_{scaled}, y_{scaled}` are intermediate coordinates in a standard plane.

   4. **Tangential Projection**: Convert to angular coordinates using the TAN projection

      .. math::
         r = \sqrt{x_{scaled}^2 + y_{scaled}^2}

         \phi = \tan^{-1}(-x_{scaled}, y_{scaled}) \cdot \frac{180}{\pi}

         \theta = \tan^{-1}(r_0, r) \cdot \frac{180}{\pi}

      where :math:`r_0 = \frac{180}{\pi}` is the radius scaling factor, :math:`\phi` is the native longitude
      and :math:`\theta` is the native latitude in the celestial coordinate system.

   5. **Sky Coordinates Conversion**: Transform to world coordinates (RA, Dec)

      .. math::
         \sin(dec) = \sin(\theta) \cdot \sin(dec_0) + \cos(\theta) \cdot \cos(dec_0) \cdot \cos(\phi)

         dec = \sin^{-1}(\sin(dec))

         ra = ra_0 + \tan^{-1}(-\cos(\theta) \cdot \sin(\phi), \sin(\theta) \cdot \cos(dec_0) - \cos(\theta) \cdot \sin(dec_0) \cdot \cos(\phi))

      where :math:`ra_0, dec_0` are the reference point sky coordinates (CRVAL1, CRVAL2), and :math:`ra, dec`
      are the final right ascension and declination in degrees.

   The implementation faithfully reproduces the WCSLib computation steps for the TAN projection,
   including precise handling of trigonometric functions and coordinate transformations.

calculate_sourceCoords
----------------------

.. py:function:: calculate_sourceCoords(self)

   Calculate source image pixel coordinates corresponding to each target image pixel.

   This function performs the inverse transformation from the target image grid through
   sky coordinates and back to pixel coordinates in the source image. This is especially
   useful for image resampling and alignment tasks.

   Returns source image pixel coordinates (x, y) corresponding to each target pixel.

   **Key Steps:**

   1. **Sky Coordinate Calculation**: Obtain sky coordinates (RA, Dec) using `calculate_skyCoords`

   2. **World to Native Transformation**: Convert world coordinates to native spherical coordinates

      .. math::
         \delta_{ra} = ra - ra_0

         y_{\phi} = -\cos(dec) \cdot \sin(\delta_{ra})

         x_{\phi} = \sin(dec) \cdot \cos(dec_0) - \cos(dec) \cdot \sin(dec_0) \cdot \cos(\delta_{ra})

         \phi = \tan^{-1}(y_{\phi}, x_{\phi}) \cdot \frac{180}{\pi}

         \theta = \sin^{-1}(\sin(dec) \cdot \sin(dec_0) + \cos(dec) \cdot \cos(dec_0) \cdot \cos(\delta_{ra})) \cdot \frac{180}{\pi}

      where :math:`ra, dec` are the input sky coordinates, :math:`ra_0, dec_0` are the reference point coordinates (CRVAL1, CRVAL2),
      :math:`\delta_{ra}` is the difference in right ascension, and :math:`\phi, \theta` are native spherical coordinates.

   3. **Projection Inversion**: Apply inverse TAN projection

      .. math::
         r_0 = \frac{180}{\pi}

         r = r_0 \cdot \frac{\cos(\theta)}{\sin(\theta)}

         x_{scaled} = -r \cdot \sin(\phi)

         y_{scaled} = r \cdot \cos(\phi)

      where :math:`r_0` is the radius scaling factor, :math:`r` is the radial distance, and :math:`x_{scaled}, y_{scaled}`
      are intermediate coordinates in the standard projection plane.

   4. **Linear Transformation**: Apply inverse CD matrix to get pixel offsets

      .. math::
         CD\_matrix = PC\_matrix \cdot CDELT

         \begin{bmatrix} u \\ v \end{bmatrix} = CD\_matrix^{-1} \cdot \begin{bmatrix} x_{scaled} \\ y_{scaled} \end{bmatrix}

      where :math:`CD\_matrix^{-1}` is the inverse of the combined transformation matrix, and :math:`u, v` are pixel
      offsets before SIP correction.

   5. **SIP Correction**: Apply inverse SIP distortion correction if present

      .. math::
         (u', v') = \text{apply_inverse_sip_distortion}(u, v)

      where :math:`u', v'` are the corrected pixel offsets after applying inverse SIP distortion.

   6. **Final Coordinates**: Add reference pixel position to get final pixel coordinates

      .. math::
         x_{pixel} = u' + (CRPIX1 - 1)

         y_{pixel} = v' + (CRPIX2 - 1)

      where :math:`CRPIX1, CRPIX2` are the reference pixel coordinates, and :math:`x_{pixel}, y_{pixel}` are
      the final source image pixel coordinates.

   This implementation carefully handles the sign conventions and coordinate system transformations
   required for accurate results, matching the behavior of the standard WCSLib implementation.

Notes
-----

- Both functions support batch processing for efficient transformation of multiple coordinates
- The code uses PyTorch tensors for GPU acceleration and differentiable operations
- Special care is taken to handle edge cases like singularities in the projections
- The implementation follows the WCSLib conventions for sign handling and coordinate transformations