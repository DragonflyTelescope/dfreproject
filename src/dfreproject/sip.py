from typing import Tuple

import astropy
import torch


def get_sip_coeffs(
    wcs: astropy.wcs.WCS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract SIP polynomial coefficients from a WCS object.

    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        WCS object potentially containing SIP distortion

    Returns:
    --------
    dict:
        Dictionary containing SIP coefficient matrices A, B, AP, BP and orders
    """
    sip_coeffs = {}

    # Check if SIP distortion is present
    sip = getattr(wcs, "sip", None)
    if sip is None:
        return None

    # Extract the SIP matrices
    sip_coeffs["a_order"] = sip.a_order
    sip_coeffs["b_order"] = sip.b_order
    sip_coeffs["a"] = sip.a
    sip_coeffs["b"] = sip.b

    # Check for inverse coefficients
    if hasattr(sip, "ap_order") and sip.ap_order > 0:
        sip_coeffs["ap_order"] = sip.ap_order
        sip_coeffs["ap"] = sip.ap
    else:
        sip_coeffs["ap_order"] = 0

    if hasattr(sip, "bp_order") and sip.bp_order > 0:
        sip_coeffs["bp_order"] = sip.bp_order
        sip_coeffs["bp"] = sip.bp
    else:
        sip_coeffs["bp_order"] = 0

    return sip_coeffs


def apply_sip_distortion(
    u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str = "cpu"
):
    """
    Apply SIP distortion to intermediate pixel coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Intermediate pixel coordinates (before distortion)
    sip_coeffs : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on

    Returns:
    --------
    tuple:
        (u', v') distorted coordinates
    """
    if sip_coeffs is None:
        return u, v

    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Get the SIP orders
    a_order = sip_coeffs["a_order"]

    # Convert coefficient matrices to tensors
    a_matrix = torch.tensor(sip_coeffs["a"], device=device)
    b_matrix = torch.tensor(sip_coeffs["b"], device=device)

    # Initialize correction terms
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    # For array inputs, reshape to make computation easier
    orig_shape = u.shape
    u_flat = u.reshape(-1) if u.dim() > 0 else u.unsqueeze(0)
    v_flat = v.reshape(-1) if v.dim() > 0 else v.unsqueeze(0)
    f_u_flat = f_u.reshape(-1) if f_u.dim() > 0 else f_u.unsqueeze(0)
    f_v_flat = f_v.reshape(-1) if f_v.dim() > 0 else f_v.unsqueeze(0)

    # Apply the polynomial distortion
    for i in range(a_order + 1):
        for j in range(a_order + 1 - i):
            if i == 0 and j == 0:
                continue  # Skip the 0,0 term

            # Compute u^i * v^j for all points
            pow_term = (u_flat**i) * (v_flat**j)

            # Apply coefficient
            f_u_flat += a_matrix[i, j] * pow_term
            f_v_flat += b_matrix[i, j] * pow_term

    # Reshape back to original shape if needed
    if u.dim() > 0:
        f_u = f_u_flat.reshape(orig_shape)
        f_v = f_v_flat.reshape(orig_shape)
    else:
        f_u = f_u_flat[0]
        f_v = f_v_flat[0]

    # Add the distortion terms to get the corrected coordinates
    u_corrected = u + f_u
    v_corrected = v + f_v

    return u_corrected, v_corrected


def apply_inverse_sip_distortion(
    u: torch.Tensor, v: torch.Tensor, sip_coeffs: Tuple, device: str = "cpu"
):
    """
    Apply inverse SIP distortion to go from distorted to intermediate coordinates.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates
    sip_coeffs : Tuple
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates
    """
    if sip_coeffs is None:
        return u, v

    # Check if inverse coefficients are available
    if sip_coeffs["ap_order"] == 0 or sip_coeffs["bp_order"] == 0:
        # Use iterative method if inverse coefficients aren't available
        return iterative_inverse_sip(u, v, sip_coeffs, device)

    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Get the SIP orders
    ap_order = sip_coeffs["ap_order"]

    # Convert coefficient matrices to tensors
    ap_matrix = torch.tensor(sip_coeffs["ap"], device=device)
    bp_matrix = torch.tensor(sip_coeffs["bp"], device=device)

    # Initialize correction terms
    f_u = torch.zeros_like(u)
    f_v = torch.zeros_like(v)

    # For array inputs, reshape to make computation easier
    orig_shape = u.shape
    u_flat = u.reshape(-1) if u.dim() > 0 else u.unsqueeze(0)
    v_flat = v.reshape(-1) if v.dim() > 0 else v.unsqueeze(0)
    f_u_flat = f_u.reshape(-1) if f_u.dim() > 0 else f_u.unsqueeze(0)
    f_v_flat = f_v.reshape(-1) if f_v.dim() > 0 else f_v.unsqueeze(0)

    # Apply the polynomial correction
    for i in range(ap_order + 1):
        for j in range(ap_order + 1 - i):
            if i == 0 and j == 0:
                continue  # Skip the 0,0 term

            # Compute u^i * v^j for all points
            pow_term = (u_flat**i) * (v_flat**j)

            # Apply coefficient
            f_u_flat += ap_matrix[i, j] * pow_term
            f_v_flat += bp_matrix[i, j] * pow_term

    # Reshape back to original shape if needed
    if u.dim() > 0:
        f_u = f_u_flat.reshape(orig_shape)
        f_v = f_v_flat.reshape(orig_shape)
    else:
        f_u = f_u_flat[0]
        f_v = f_v_flat[0]

    # Add the correction terms to get the undistorted coordinates
    u_corrected = u + f_u
    v_corrected = v + f_v

    return u_corrected, v_corrected


def iterative_inverse_sip(
    u: torch.Tensor,
    v: torch.Tensor,
    sip_coeffs: Tuple,
    device: str = "cpu",
    max_iter: int = 20,
    tol: float = 1e-8,
):
    """
    Iteratively solve for undistorted coordinates when inverse SIP coefficients
    are not available.

    Parameters:
    -----------
    u, v : torch.Tensor
        Distorted coordinates
    sip_coeffs : Tuple
        SIP coefficient matrices
    device : torch.device, optional
        Device to place tensors on
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance

    Returns:
    --------
    tuple:
        (u', v') undistorted coordinates
    """
    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=device)
        v = torch.tensor(v, device=device)

    # Initial guess: undistorted = distorted
    u_undist = u.clone()
    v_undist = v.clone()

    for i in range(max_iter):
        # Apply forward SIP to get predicted distorted coordinates
        u_pred, v_pred = apply_sip_distortion(u_undist, v_undist, sip_coeffs, device)

        # Compute error
        u_error = u - u_pred
        v_error = v - v_pred

        # Check convergence
        max_error = torch.max(
            torch.abs(torch.cat([u_error.flatten(), v_error.flatten()]))
        )
        if max_error < tol:
            break

        # Update undistorted coordinates
        u_undist = u_undist + u_error
        v_undist = v_undist + v_error

    return u_undist, v_undist
