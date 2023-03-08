import copy
from functools import partial
from typing import Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch as T
import wandb
from jetnet.evaluation import w1efp, w1m, w1p

from src.models.diffusion import VPDiffusionSchedule, run_sampler
from src.models.modules import CosineEncoding, IterativeNormLayer
from src.models.schedulers import WarmupToConstant
from src.models.transformers import FullTransformerEncoder
from src.numpy_utils import undo_log_squash
from src.plotting import plot_mpgan_marginals
from src.torch_utils import get_loss_fn, to_np


class TransformerDiffusionGenerator(pl.LightningModule):
    """A generative model which uses the diffusion process on a point cloud."""

    def __init__(
        self,
        *,
        pc_dim: list,
        ctxt_dim: int,
        n_nodes: int,
        cosine_config: Mapping,
        diff_config: Mapping,
        normaliser_config: Mapping,
        trans_enc_config: Mapping,
        optimizer: partial,
        loss_name: str = "mse",
        mle_loss_weight: float = 0.0,
        ema_sync: float = 0.999,
        sampler_name: str = "em",
        sampler_steps: int = 100,
    ) -> None:
        """
        Args:
            pc_dim: The dimension of the point cloud
            ctxt_dim: The size of the context vector for the point cloud
            n_nodes: Max number of nodes used to train this model
            cosine_config: For defining the cosine embedding arguments
            normaliser_config: For defining the iterative normalisation layer
            diff_shedule: The diffusion scheduler, defines the signal and noise rates
            trans_enc_config: Keyword arguments for the TransformerEncoder network
            optimizer: Partially initialised optimizer
            sched_config: The config for how to apply the scheduler
            ema_sync: How fast the ema network syncs with the given one
            loss_name: Name of the loss function to use for noise estimation
            mle_loss_weight: Relative weight of the Maximum-Liklihood loss term
            sampler_name: Name of O/SDE solver, does not effect training.
            sampler_steps: Steps used in generation, does not effect training.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Class attributes
        self.pc_dim = pc_dim
        self.ctxt_dim = ctxt_dim
        self.n_nodes = n_nodes
        self.loss_fn = get_loss_fn(loss_name)
        self.mle_loss_weight = mle_loss_weight
        self.ema_sync = ema_sync

        # The encoder and scheduler needed for diffusion
        self.diff_sched = VPDiffusionSchedule(**diff_config)
        self.time_encoder = CosineEncoding(**cosine_config)

        # The layer which normalises the input point cloud data
        self.normaliser = IterativeNormLayer((pc_dim,), **normaliser_config)
        if self.ctxt_dim:
            self.ctxt_normaliser = IterativeNormLayer((ctxt_dim,), **normaliser_config)

        # The denoising transformer
        self.net = FullTransformerEncoder(
            inpt_dim=pc_dim,
            outp_dim=pc_dim,
            ctxt_dim=ctxt_dim + self.time_encoder.outp_dim,
            **trans_enc_config,
        )

        # A copy of the network which will sync with an exponential moving average
        self.ema_net = copy.deepcopy(self.net)

        # Sampler to run in the validation/testing loop
        self.sampler_name = sampler_name
        self.sampler_steps = sampler_steps

        # Record of the outputs of the validation step
        self.val_outs = []

    def forward(
        self,
        noisy_data: T.Tensor,
        diffusion_times: T.Tensor,
        mask: T.BoolTensor,
        ctxt: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """Pass through the model and get an estimate of the noise added to the
        input."""

        # Use the appropriate network for training or validation
        if self.training:
            network = self.net
        else:
            network = self.ema_net

        # Encode the times and combine with existing context info
        context = self.time_encoder(diffusion_times)
        if self.ctxt_dim:
            context = T.cat([context, ctxt], dim=-1)

        # Use the selected network to esitmate the noise present in the data
        return network(noisy_data, mask=mask, ctxt=context)

    def _shared_step(self, sample: tuple) -> Tuple[T.Tensor, T.Tensor]:
        """Shared step used in both training and validaiton."""

        # Unpack the sample tuple
        nodes, mask, ctxt = sample

        # Pass through the normalisers
        nodes = self.normaliser(nodes, mask)
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)

        # Sample from the gaussian latent space to perturb the point clouds
        noises = T.randn_like(nodes) * mask.unsqueeze(-1)

        # Sample uniform random diffusion times and get the rates
        diffusion_times = T.rand(size=(len(nodes), 1), device=self.device)
        signal_rates, noise_rates = self.diff_sched(diffusion_times.view(-1, 1, 1))

        # Mix the signal and noise according to the diffusion equation
        noisy_nodes = signal_rates * nodes + noise_rates * noises

        # Predict the noise using the network
        pred_noises = self.forward(noisy_nodes, diffusion_times, mask, ctxt)

        # Simple noise loss is for "perceptual quality"
        simple_loss = self.loss_fn(noises[mask], pred_noises[mask])

        # MLE loss is for maximum liklihood training
        if self.mle_loss_weight:
            betas = self.diff_sched.get_betas(diffusion_times.view(-1, 1, 1))
            mle_weights = betas / noise_rates
            mle_loss = mle_weights * simple_loss
        else:
            mle_loss = T.zeros_like(simple_loss)

        return simple_loss.mean(), mle_loss.mean()

    def training_step(self, sample: tuple, _batch_idx: int) -> T.Tensor:
        simple_loss, mle_loss = self._shared_step(sample)
        total_loss = simple_loss + self.mle_loss_weight * mle_loss
        self.log("train/simple_loss", simple_loss)
        self.log("train/mle_loss", mle_loss)
        self.log("train/total_loss", total_loss)
        self._sync_ema_network()
        return total_loss

    def validation_step(self, sample: tuple, batch_idx: int) -> None:
        simple_loss, mle_loss = self._shared_step(sample)
        total_loss = simple_loss + self.mle_loss_weight * mle_loss
        self.log("valid/simple_loss", simple_loss)
        self.log("valid/mle_loss", mle_loss)
        self.log("valid/total_loss", total_loss)

        # Run the full generation of the sample during a validation step
        outputs = self.full_generation(
            self.sampler_name,
            self.sampler_steps,
            mask=sample[1],
            ctxt=sample[2],
        )

        # Add to the collection of the validaiton outputs
        self.val_outs.append((to_np(outputs), to_np(sample)))

    def on_validation_epoch_end(self) -> None:
        """At the end of the validation epoch, calculate and log the metrics
        and plot the histograms.

        This function right now only works with MPGAN configs
        """

        # Combine all outputs
        gen_nodes = np.vstack([v[0] for v in self.val_outs])
        real_nodes = np.vstack([v[1][0] for v in self.val_outs])
        mask = np.vstack([v[1][1] for v in self.val_outs])
        high = np.vstack([v[1][2] for v in self.val_outs])

        # Change the data from log(pt+1) into pt fraction (needed for metrics)
        if self.trainer.datamodule.hparams.data_conf.log_squash_pt:
            gen_nodes[..., -1] = undo_log_squash(gen_nodes[..., -1]) / high[..., 0:1]
            real_nodes[..., -1] = undo_log_squash(real_nodes[..., -1]) / high[..., 0:1]

        # Apply clipping
        gen_nodes = np.nan_to_num(gen_nodes)
        gen_nodes[..., 0] = np.clip(gen_nodes[..., 0], -0.5, 0.5)
        gen_nodes[..., 1] = np.clip(gen_nodes[..., 1], -0.5, 0.5)
        gen_nodes[..., 2] = np.clip(gen_nodes[..., 2], 0, 1)
        real_nodes = np.nan_to_num(real_nodes)
        real_nodes[..., 0] = np.clip(real_nodes[..., 0], -0.5, 0.5)
        real_nodes[..., 1] = np.clip(real_nodes[..., 1], -0.5, 0.5)
        real_nodes[..., 2] = np.clip(real_nodes[..., 2], 0, 1)

        # Calculate and log the Wasserstein discriminants
        bootstrap = {
            "num_eval_samples": 10000,
            "num_batches": 10,
        }
        w1m_val, w1m_err = w1m(real_nodes, gen_nodes, **bootstrap)
        w1p_val, w1p_err = w1p(real_nodes, gen_nodes, **bootstrap)
        w1efp_val, w1efp_err = w1efp(real_nodes, gen_nodes, efp_jobs=1, **bootstrap)
        self.log("valid/w1m", w1m_val)
        self.log("valid/w1m_err", w1m_err)
        self.log("valid/w1p", w1p_val.mean())
        self.log("valid/w1p_err", w1p_err.mean())
        self.log("valid/w1efp", w1efp_val.mean())
        self.log("valid/w1efp_err", w1efp_err.mean())

        # Plot the MPGAN-like marginals
        plot_mpgan_marginals(gen_nodes, real_nodes, mask, self.trainer.current_epoch)
        self.val_outs.clear()

    def _sync_ema_network(self) -> None:
        """Updates the Exponential Moving Average Network."""
        with T.no_grad():
            for params, ema_params in zip(
                self.net.parameters(), self.ema_net.parameters()
            ):
                ema_params.data.copy_(
                    self.ema_sync * ema_params.data
                    + (1.0 - self.ema_sync) * params.data
                )

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""

        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            wandb.define_metric("train/simple_loss", summary="min")
            wandb.define_metric("train/mle_loss", summary="min")
            wandb.define_metric("train/total_loss", summary="min")
            wandb.define_metric("valid/simple_loss", summary="min")
            wandb.define_metric("valid/mle_loss", summary="min")
            wandb.define_metric("valid/total_loss", summary="min")
            wandb.define_metric("valid/w1m", summary="min")
            wandb.define_metric("valid/w1p", summary="min")
            wandb.define_metric("valid/w1efp", summary="min")

    def set_sampler(
        self, sampler_name: Optional[str] = None, sampler_steps: Optional[int] = None
    ) -> None:
        """Replaces the sampler list with a new one."""
        if sampler_name is not None:
            self.sampler_name = sampler_name
        if sampler_steps is not None:
            self.sampler_steps = sampler_steps

    def full_generation(
        self,
        sampler: str,
        steps: int,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        initial_noise: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """Fully generate a batch of data from noise, given context information
        and a mask."""

        # Either a mask or initial noise must be defined or we dont know how
        # many samples to generate and with what cardinality
        if mask is None and initial_noise is None:
            raise ValueError("Please provide either a mask or noise to generate from")
        if mask is None:
            mask = T.full(initial_noise.shape[:-1], True, device=self.device)
        if initial_noise is None:
            initial_noise = T.randn((*mask.shape, self.pc_dim), device=self.device)

        # Normalise the context
        if self.ctxt_dim:
            ctxt = self.ctxt_normaliser(ctxt)
            assert len(ctxt) == len(initial_noise)

        # Run the sampling method
        outputs, _ = run_sampler(
            sampler,
            self,
            self.diff_sched,
            initial_noise=initial_noise * mask.unsqueeze(-1),
            n_steps=steps,
            mask=mask,
            ctxt=ctxt,
            clip_predictions=(-25, 25),
        )

        # Ensure that the output adheres to the mask
        outputs[~mask] = 0

        # Return the normalisation of the generated point cloud
        return self.normaliser.reverse(outputs, mask=mask)

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""

        # Finish initialising the optimiser and create the scheduler
        opt = self.hparams.optimizer(params=self.parameters())
        sched = WarmupToConstant(opt, num_steps=10_000)

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }
