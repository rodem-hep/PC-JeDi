import math
from typing import Optional, Tuple

import torch as T
from tqdm import tqdm


class VPDiffusionSchedule:
    def __init__(self, max_sr: float = 1, min_sr: float = 1e-2) -> None:
        self.max_sr = max_sr
        self.min_sr = min_sr

    def __call__(self, time: T.Tensor) -> T.Tensor:
        return cosine_diffusion_shedule(time, self.max_sr, self.min_sr)

    def get_betas(self, time: T.Tensor) -> T.Tensor:
        return cosine_beta_shedule(time, self.max_sr, self.min_sr)


def cosine_diffusion_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-2
) -> Tuple[T.Tensor, T.Tensor]:
    """Calculates the signal and noise rate for any point in the diffusion
    processes.

    Using continuous diffusion times between 0 and 1 which make switching between
    different numbers of diffusion steps between training and testing much easier.
    Returns only the values needed for the jump forward diffusion step and the reverse
    DDIM step.
    These are sqrt(alpha_bar) and sqrt(1-alphabar) which are called the signal_rate
    and noise_rate respectively.

    The jump forward diffusion process is simply a weighted sum of:
        input * signal_rate + eps * noise_rate

    Uses a cosine annealing schedule as proposed in
    Proposed in https://arxiv.org/abs/2102.09672

    Args:
        diff_time: The time used to sample the diffusion scheduler
            Output will match the shape
            Must be between 0 and 1
        max_sr: The initial rate at the first step
        min_sr: How much signal is preserved at end of diffusion
            (can't be zero due to log)
    """

    # Use cosine annealing, which requires switching from times -> angles
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    signal_rates = T.cos(diffusion_angles)
    noise_rates = T.sin(diffusion_angles)
    return signal_rates, noise_rates


def cosine_beta_shedule(
    diff_time: T.Tensor, max_sr: float = 1, min_sr: float = 1e-2
) -> T.Tensor:
    """Returns the beta values for the continuous flows using the above cosine
    scheduler."""
    start_angle = math.acos(max_sr)
    end_angle = math.acos(min_sr)
    diffusion_angles = start_angle + diff_time * (end_angle - start_angle)
    return 2 * (end_angle - start_angle) * T.tan(diffusion_angles)


def ddim_predict(
    noisy_data: T.Tensor,
    pred_noises: T.Tensor,
    signal_rates: T.Tensor,
    noise_rates: T.Tensor,
) -> T.Tensor:
    """Use a single ddim step to predict the final image from anywhere in the
    diffusion process."""
    return (noisy_data - noise_rates * pred_noises) / signal_rates


@T.no_grad()
def ddim_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: T.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[T.Tensor] = None,
    ctxt: Optional[T.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
) -> Tuple[T.Tensor, list]:
    """Apply the DDIM sampling process to generate a batch of samples from
    noise.

    Args:
        model: A denoising diffusion model
            Requires: inpt_dim, device, forward() method that outputs pred noise
        diif_sched: A diffusion schedule object to calculate signal and noise rates
        initial_noise: The initial noise to pass through the process
            If none it will be generated here
        n_steps: The number of iterations to generate the samples
        keep_all: Return all stages of diffusion process
            Can be memory heavy for large batches
        num_samples: How many samples to generate
            Ignored if initial_noise is provided
        mask: The mask for the output point clouds
        ctxt: The context tensor for the output point clouds
        clip_predictions: Can stabalise generation by clipping the outputs
    """

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # The shape needed for expanding the time encodings
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    step_size = 1 / n_steps

    # The initial variables needed for the loop
    noisy_data = initial_noise
    diff_times = T.ones(num_samples, device=model.device)
    next_signal_rates, next_noise_rates = diff_sched(diff_times.view(expanded_shape))
    for step in tqdm(range(n_steps), "DDIM-sampling", leave=False):

        # Update with the previous 'next' step
        signal_rates = next_signal_rates
        noise_rates = next_noise_rates

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(noisy_data)

        # Apply the denoise step to get X_0 and expected noise
        pred_noises = model(noisy_data, diff_times, mask, ctxt)
        pred_data = ddim_predict(noisy_data, pred_noises, signal_rates, noise_rates)

        # Get the next predicted components using the next signal and noise rates
        diff_times = diff_times - step_size
        next_signal_rates, next_noise_rates = diff_sched(
            diff_times.view(expanded_shape)
        )

        # Clamp the predicted X_0 for stability
        if clip_predictions is not None:
            pred_data.clamp_(*clip_predictions)

        # Remix the predicted components to go from estimated X_0 -> X_{t-1}
        noisy_data = next_signal_rates * pred_data + next_noise_rates * pred_noises

    return pred_data, all_stages


@T.no_grad()
def euler_maruyama_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: T.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[T.Tensor] = None,
    ctxt: Optional[T.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
) -> Tuple[T.Tensor, list]:
    """Apply the full reverse process to noise to generate a batch of
    samples."""

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # The shape needed for expanding the time encodings
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    delta_t = 1 / n_steps

    # The initial variables needed for the loop
    x_t = initial_noise
    t = T.ones(num_samples, device=model.device)
    for step in tqdm(range(n_steps), "Euler-Maruyama-sampling", leave=False):

        # Use the model to get the expected noise
        pred_noises = model(x_t, t, mask, ctxt)

        # Use to get s_theta
        _, noise_rates = diff_sched(t.view(expanded_shape))
        s = -pred_noises / noise_rates

        # Take one step using the em method
        betas = diff_sched.get_betas(t.view(expanded_shape))
        x_t += 0.5 * betas * (x_t + 2 * s) * delta_t
        x_t += (betas * delta_t).sqrt() * T.randn_like(x_t)
        t -= delta_t

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(x_t)

        # Clamp the denoised data for stability
        if clip_predictions is not None:
            x_t.clamp_(*clip_predictions)

    return x_t, all_stages


@T.no_grad()
def euler_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: T.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[T.Tensor] = None,
    ctxt: Optional[T.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
) -> Tuple[T.Tensor, list]:
    """Apply the full reverse process to noise to generate a batch of
    samples."""

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # The shape needed for expanding the time encodings
    expanded_shape = [-1] + [1] * (initial_noise.dim() - 1)

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    delta_t = 1 / n_steps

    # The initial variables needed for the loop
    t = T.ones(num_samples, device=model.device)
    signal_rates, noise_rates = diff_sched(t.view(expanded_shape))
    x_t = initial_noise * (signal_rates + noise_rates)
    for step in tqdm(range(n_steps), "Euler-sampling", leave=False):

        # Take a step using the euler method and the gradient calculated by the ode
        x_t += get_ode_gradient(model, diff_sched, x_t, t, mask, ctxt) * delta_t
        t -= delta_t

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(x_t)

        # Clamp the denoised data for stability
        if clip_predictions is not None:
            x_t.clamp_(*clip_predictions)

    return x_t, all_stages


@T.no_grad()
def runge_kutta_sampler(
    model,
    diff_sched: VPDiffusionSchedule,
    initial_noise: T.Tensor,
    n_steps: int = 50,
    keep_all: bool = False,
    mask: Optional[T.Tensor] = None,
    ctxt: Optional[T.BoolTensor] = None,
    clip_predictions: Optional[tuple] = None,
) -> Tuple[T.Tensor, list]:
    """Apply the full reverse process to noise to generate a batch of
    samples."""

    # Get the initial noise for generation and the number of sammples
    num_samples = initial_noise.shape[0]

    # Check the input argument for the n_steps, must be less than what was trained
    all_stages = []
    delta_t = 1 / n_steps

    # Wrap the ode gradient in a lambda function depending only on xt and t
    ode_grad = lambda t, x_t: get_ode_gradient(model, diff_sched, x_t, t, mask, ctxt)

    # The initial variables needed for the loop
    x_t = initial_noise
    t = T.ones(num_samples, device=model.device)
    for step in tqdm(range(n_steps), "Runge-Kutta-sampling", leave=False):

        k1 = delta_t * (ode_grad(t, x_t))
        k2 = delta_t * (ode_grad((t - delta_t / 2), (x_t + k1 / 2)))
        k3 = delta_t * (ode_grad((t - delta_t / 2), (x_t + k2 / 2)))
        k4 = delta_t * (ode_grad((T.clamp_min(t - delta_t, 0)), (x_t + k3)))
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_t += k
        t -= delta_t

        # Keep track of the diffusion evolution
        if keep_all:
            all_stages.append(x_t)

        # Clamp the denoised data for stability
        if clip_predictions is not None:
            x_t.clamp_(*clip_predictions)

    return x_t, all_stages


def get_ode_gradient(
    model,
    diff_sched: VPDiffusionSchedule,
    x_t: T.Tensor,
    t: T.Tensor,
    mask: Optional[T.BoolTensor] = None,
    ctxt: Optional[T.Tensor] = None,
) -> T.Tensor:
    expanded_shape = [-1] + [1] * (x_t.dim() - 1)
    _, noise_rates = diff_sched(t.view(expanded_shape))
    betas = diff_sched.get_betas(t.view(expanded_shape))
    return 0.5 * betas * (x_t - model(x_t, t, mask, ctxt) / noise_rates)


def run_sampler(sampler: str, *args, **kwargs) -> Tuple[T.Tensor, list]:
    if sampler == "em":
        return euler_maruyama_sampler(*args, **kwargs)
    if sampler == "euler":
        return euler_sampler(*args, **kwargs)
    if sampler == "rk":
        return runge_kutta_sampler(*args, **kwargs)
    if sampler == "ddim":
        return ddim_sampler(*args, **kwargs)
    raise RuntimeError(f"Unknown sampler: {sampler}")
