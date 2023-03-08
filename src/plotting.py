from pathlib import Path

import numpy as np
import wandb
from jetnet.utils import efps
from mattstools.mattstools.plotting import plot_multi_hists


def locals_to_rel_mass_and_efp(csts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert the values of a set of constituents to the relative mass and EFP
    values of the jet they belong to.

    Args:
        csts: A numpy array of shape (batch_size, n_csts, 3)
            containing the (eta, phi, pt) values of the constituents.
        mask: A numpy array of shape (batch_size, n_csts)
            containing a mask for the constituents, used to sum only over
            the valid constituents.

    Returns:
        A numpy array of shape (batch_size, 2)
            containing the relative mass and EFP values of the jet.
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_m = np.sqrt(
        np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None)
    )

    # Get the efp values
    jet_efps = efps(csts, efp_jobs=1).mean(axis=-1)

    return np.vstack([jet_m, jet_efps]).T


def plot_mpgan_marginals(
    outputs: np.ndarray,
    nodes: np.ndarray,
    mask: np.ndarray,
    current_epoch: int,
) -> None:

    # Clip the outputs for the marginals to match expected jet spread
    outputs[..., 0] = np.clip(outputs[..., 0], -0.5, 0.5)
    outputs[..., 1] = np.clip(outputs[..., 1], -0.5, 0.5)
    outputs[..., 2] = np.clip(outputs[..., 2], 0, 1)

    # Plot histograms for the constituent marginals
    Path("./plots/").mkdir(parents=False, exist_ok=True)
    cst_img = plot_multi_hists(
        data_list=[nodes[mask], outputs[mask]],
        type_labels=["Original", "Generated"],
        col_labels=[r"$\Delta \eta$", r"$\Delta \phi$", r"$\frac{p_T}{Jet_{p_T}}$"],
        normed=True,
        return_img=True,
        path=f"./plots/csts_{current_epoch}",
        logy=True,
    )

    # Convert to total jet mass and pt, do some clamping to make everyone happy
    pred_jets = locals_to_rel_mass_and_efp(outputs, mask)
    pred_jets[:, 0] = np.clip(pred_jets[:, 0], 0, 0.4)
    pred_jets[:, 1] = np.clip(pred_jets[:, 1], 0, 4e-3)
    pred_jets = np.nan_to_num(pred_jets)

    real_jets = locals_to_rel_mass_and_efp(nodes, mask)
    real_jets[:, 0] = np.clip(real_jets[:, 0], 0, 0.4)
    real_jets[:, 1] = np.clip(real_jets[:, 1], 0, 4e-3)
    real_jets = np.nan_to_num(real_jets)

    # Image for the total jet variables
    jet_img = plot_multi_hists(
        data_list=[real_jets, pred_jets],
        type_labels=["Original", "Generated"],
        col_labels=["Relative Jet Mass", "Jet EFP"],
        normed=True,
        return_img=True,
        path=f"./plots/jets_{current_epoch}",
    )

    # Create the wandb table and add the data
    if wandb.run is not None:
        gen_table = wandb.Table(columns=["constituents", "jets"])
        gen_table.add_data(wandb.Image(cst_img), wandb.Image(jet_img))
        wandb.run.log({"generated": gen_table}, commit=False)
