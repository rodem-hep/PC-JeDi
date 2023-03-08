# import jetnet
import numpy as np
import pytorch_lightning as pl
import torch as T

# FIX RANDOM SEED FOR REPRODUCIBILITY
pl.seed_everything(0, workers=True)


def locals_to_mass_and_pt(csts: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - del_eta
    - del_phi
    - log_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = csts[..., 2].exp()

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.clamp_min(jet_px**2 + jet_py**2, 0).sqrt()
    jet_m = T.clamp_min(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0).sqrt()

    return T.vstack([jet_pt, jet_m]).T


def numpy_locals_to_mass_and_pt(
    csts: np.ndarray,
    mask: np.ndarray,
    pt_logged=False,
) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - del_eta
    - del_phi
    - log_pt or just pt depending on pt_logged
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = np.exp(csts[..., 2]) * mask if pt_logged else csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(
        np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None)
    )

    return np.vstack([jet_pt, jet_m]).T
