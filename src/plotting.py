from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import PIL
import wandb
from jetnet.utils import efps


def plot_multi_hists(
    data_list: Union[list, np.ndarray],
    data_labels: Union[list, str],
    col_labels: Union[list, str],
    path: Optional[Union[Path, str]] = None,
    scale_factors: Optional[list] = None,
    do_err: bool = False,
    do_norm: bool = False,
    bins: Union[list, str, partial] = "auto",
    logy: bool = False,
    y_label: Optional[str] = None,
    ylim: Optional[list] = None,
    rat_ylim: tuple = (0, 2),
    rat_label: Optional[str] = None,
    scale: int = 5,
    do_legend: bool = True,
    hist_kwargs: Optional[list] = None,
    err_kwargs: Optional[list] = None,
    legend_kwargs: Optional[dict] = None,
    incl_overflow: bool = True,
    incl_underflow: bool = True,
    do_ratio_to_first: bool = False,
    return_fig: bool = False,
    return_img: bool = False,
) -> Union[plt.Figure, None]:
    """Plot multiple histograms given a list of 2D tensors/arrays.

    - Performs the histogramming here
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis
    - If the tensor being passed is 3D it will average them and combine the uncertainty

    args:
        data_list: A list of tensors or numpy arrays, each col will be a seperate axis
        data_labels: A list of labels for each tensor in data_list
        col_labels: A list of labels for each column/axis
        path: The save location of the plots (include img type)
        scale_factors: List of scalars to be applied to each histogram
        do_err: If the statistical errors should be included as shaded regions
        do_norm: If the histograms are to be a density plot
        bins: List of bins to use for each axis, can use numpy's strings
        logy: If we should use the log in the y-axis
        y_label: Label for the y axis of the plots
        ylim: The y limits for all plots
        rat_ylim: The y limits of the ratio plots
        rat_label: The label for the ratio plot
        scale: The size in inches for each subplot
        do_legend: If the legend should be plotted
        hist_kwargs: Additional keyword arguments for the line for each histogram
        legend_kwargs: Extra keyword arguments to pass to the legend constructor
        incl_overflow: Have the final bin include the overflow
        incl_underflow: Have the first bin include the underflow
        do_ratio_to_first: Include a ratio plot to the first histogram in the list
        as_pdf: Also save an additional image in pdf format
        return_fig: Return the figure (DOES NOT CLOSE IT!)
        return_img: Return a PIL image (will close the figure)
    """

    # Make the arguments lists for generality
    if not isinstance(data_list, list):
        data_list = [data_list]
    if isinstance(data_labels, str):
        data_labels = [data_labels]
    if isinstance(col_labels, str):
        col_labels = [col_labels]
    if not isinstance(bins, list):
        bins = data_list[0].shape[-1] * [bins]
    if not isinstance(scale_factors, list):
        scale_factors = len(data_list) * [scale_factors]
    if not isinstance(hist_kwargs, list):
        hist_kwargs = len(data_list) * [hist_kwargs]
    if not isinstance(err_kwargs, list):
        err_kwargs = len(data_list) * [err_kwargs]

    # Cycle through the datalist and ensure that they are 2D, as each column is an axis
    for data_idx in range(len(data_list)):
        if data_list[data_idx].ndim < 2:
            data_list[data_idx] = data_list[data_idx].unsqueeze(-1)

    # Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = data_list[0].shape[-1]

    # Make sure that all the list lengths are consistant
    assert len(data_labels) == n_data
    assert len(col_labels) == n_axis
    assert len(bins) == n_axis

    # Make sure the there are not too many subplots
    if n_axis > 20:
        raise RuntimeError("You are asking to create more than 20 subplots!")

    # Create the figure and axes lists
    dims = np.array([1, n_axis])  # Subplot is (n_rows, n_columns)
    size = np.array([n_axis, 1.0])  # Size is (width, height)
    if do_ratio_to_first:
        dims *= np.array([2, 1])  # Double the number of rows
        size *= np.array([1, 1.2])  # Increase the height
    fig, axes = plt.subplots(
        *dims,
        figsize=tuple(scale * size),
        gridspec_kw={"height_ratios": [3, 1] if do_ratio_to_first else {1}},
        squeeze=False,
    )

    # Cycle through each axis and determine the bins that should be used
    # Automatic/Interger bins are replaced using the first item in the data list
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]
        if isinstance(ax_bins, partial):
            ax_bins = ax_bins()

        # If the axis bins was specified to be 'auto' or another numpy string
        if isinstance(ax_bins, str):
            unq = np.unique(data_list[0][:, ax_idx])
            n_unique = len(unq)

            # If the number of datapoints is less than 10 then use even spacing
            if 1 < n_unique < 10:
                ax_bins = (unq[1:] + unq[:-1]) / 2  # Use midpoints, add final, initial
                ax_bins = np.append(ax_bins, unq.max() + unq.max() - ax_bins[-1])
                ax_bins = np.insert(ax_bins, 0, unq.min() + unq.min() - ax_bins[0])

        # Numpy function to get the bin edges, catches all other cases (int, etc)
        ax_bins = np.histogram_bin_edges(data_list[0][:, ax_idx], bins=ax_bins)

        # Replace the element in the array with the edges
        bins[ax_idx] = ax_bins

    # Cycle through each of the axes
    for ax_idx in range(n_axis):

        # Get the bins for this axis
        ax_bins = bins[ax_idx]

        # Cycle through each of the data arrays
        for data_idx in range(n_data):

            # Apply overflow and underflow (make a copy)
            data = np.copy(data_list[data_idx][..., ax_idx]).squeeze()
            if incl_overflow:
                data = np.minimum(data, ax_bins[-1])
            if incl_underflow:
                data = np.maximum(data, ax_bins[0])

            # If the data is still a 2D tensor treat it as a collection of histograms
            if data.ndim > 1:
                h = []
                for dim in range(data.shape[-1]):
                    h.append(np.histogram(data[:, dim], ax_bins, density=do_norm)[0])

                # Nominal and err is based on chi2 of same value, mult measurements
                hist = 1 / np.mean(1 / np.array(h), axis=0)
                hist_err = np.sqrt(1 / np.sum(1 / np.array(h), axis=0))

            # Otherwise just calculate a single histogram
            else:
                hist, _ = np.histogram(data, ax_bins, density=do_norm)
                hist_err = np.sqrt(hist)

            # Apply the scale factors
            if scale_factors[data_idx] is not None:
                hist *= scale_factors
                hist_err *= scale_factors

            # Save the first histogram for the ratio plots
            if data_idx == 0:
                denom_hist = hist
                denom_err = hist_err

            # Get the additional keyword arguments for the histograms and errors
            if hist_kwargs[data_idx] is not None and bool(hist_kwargs[data_idx]):
                h_kwargs = deepcopy(hist_kwargs[data_idx])
            else:
                h_kwargs = {}

            # Use the stair function to plot the histograms
            line = axes[0, ax_idx].stairs(
                hist, ax_bins, label=data_labels[data_idx], **h_kwargs
            )

            if err_kwargs[data_idx] is not None and bool(err_kwargs[data_idx]):
                e_kwargs = deepcopy(err_kwargs[data_idx])
            else:
                e_kwargs = {"color": line._edgecolor, "alpha": 0.2, "fill": True}

            # Include the uncertainty in the plots as a shaded region
            if do_err:
                axes[0, ax_idx].stairs(
                    hist + hist_err,
                    ax_bins,
                    baseline=hist - hist_err,
                    **e_kwargs,
                )

            # Add a ratio plot
            if do_ratio_to_first:

                if hist_kwargs[data_idx] is not None and bool(hist_kwargs[data_idx]):
                    ratio_kwargs = deepcopy(hist_kwargs[data_idx])
                else:
                    ratio_kwargs = {
                        "color": line._edgecolor,
                        "linestyle": line._linestyle,
                    }
                ratio_kwargs["fill"] = False  # Never fill a ratio plot

                # Calculate the new ratio values with their errors
                rat_hist = hist / denom_hist
                rat_err = rat_hist * np.sqrt(
                    (hist_err / hist) ** 2 + (denom_err / denom_hist) ** 2
                )

                # Plot the ratios
                axes[1, ax_idx].stairs(
                    rat_hist,
                    ax_bins,
                    **ratio_kwargs,
                )

                # Use a standard shaded region for the errors
                if do_err:
                    axes[1, ax_idx].stairs(
                        rat_hist + rat_err,
                        ax_bins,
                        baseline=rat_hist - rat_err,
                        **e_kwargs,
                    )

    # Cycle again through each axis and apply editing
    for ax_idx in range(n_axis):
        ax_bins = bins[ax_idx]

        # X axis
        axes[0, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])
        if do_ratio_to_first:
            axes[0, ax_idx].set_xticklabels([])
            axes[1, ax_idx].set_xlabel(col_labels[ax_idx])
            axes[1, ax_idx].set_xlim(ax_bins[0], ax_bins[-1])
        else:
            axes[0, ax_idx].set_xlabel(col_labels[ax_idx])

        # Y axis
        if logy:
            axes[0, ax_idx].set_yscale("log")
        if ylim is not None:
            axes[0, ax_idx].set_ylim(*ylim)
        else:
            _, ylim2 = axes[0, ax_idx].get_ylim()
            if logy:
                axes[0, ax_idx].set_ylim(top=10 ** (np.log10(ylim2) * 1.40))
            else:
                axes[0, ax_idx].set_ylim(top=ylim2 * 1.35)
        if y_label is not None:
            axes[0, ax_idx].set_ylabel(y_label)
        elif do_norm:
            axes[0, ax_idx].set_ylabel("Normalised Entries")
        else:
            axes[0, ax_idx].set_ylabel("Entries")

        # Ratio Y axis
        if do_ratio_to_first:
            axes[1, ax_idx].set_ylim(rat_ylim)
            if rat_label is not None:
                axes[1, ax_idx].set_ylabel(rat_label)
            else:
                axes[1, ax_idx].set_ylabel(f"Ratio to {data_labels[0]}")

        # Legend
        if do_legend:
            legend_kwargs = legend_kwargs or {}
            axes[0, ax_idx].legend(**legend_kwargs)

    # Final figure layout
    fig.tight_layout()
    if do_ratio_to_first:
        fig.subplots_adjust(hspace=0.08)  # For ratio plots minimise the h_space

    # Save the file
    if path is not None:
        fig.savefig(path)

    # Return a rendered image, or the matplotlib figure, or close
    if return_img:
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close(fig)
        return img
    if return_fig:
        return fig
    plt.close(fig)


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
        data_labels=["Original", "Generated"],
        col_labels=[r"$\Delta \eta$", r"$\Delta \phi$", r"$\frac{p_T}{Jet_{p_T}}$"],
        do_norm=True,
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
        data_labels=["Original", "Generated"],
        col_labels=["Relative Jet Mass", "Jet EFP"],
        do_norm=True,
        return_img=True,
        path=f"./plots/jets_{current_epoch}",
    )

    # Create the wandb table and add the data
    if wandb.run is not None:
        gen_table = wandb.Table(columns=["constituents", "jets"])
        gen_table.add_data(wandb.Image(cst_img), wandb.Image(jet_img))
        wandb.run.log({"generated": gen_table}, commit=False)
