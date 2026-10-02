import torch

from fluxrt.flow_upscaler.upscaler_unet import UpscalerUNet
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class FlowUpscalerPipeline:
    def __init__(
        self, upscaler_unet: UpscalerUNet, scheduler: FlowMatchEulerDiscreteScheduler
    ):
        self.upscaler_unet = upscaler_unet
        self.scheduler = scheduler
        # (steps, device, dtype) -> [(timestep, dt)]. The schedule is the same
        # on every frame; scheduler.set_timesteps + scheduler.step per frame
        # cost CPU time and a GPU sync (the step looks its index up with .item()).
        self._schedules = {}

    def _schedule(self, num_inference_steps: int, device, dtype):
        key = (num_inference_steps, device, dtype)
        if key not in self._schedules:
            self.scheduler.set_timesteps(num_inference_steps, mu=1.0)
            sigmas = self.scheduler.sigmas
            self._schedules[key] = [
                (t.to(device, dtype).view(1), sigmas[i + 1] - sigmas[i])
                for i, t in enumerate(self.scheduler.timesteps)
            ]
        return self._schedules[key]

    def __call__(
        self,
        latents_small: torch.Tensor,
        target_latent_height: int | None = None,
        target_latent_width: int | None = None,
        num_inference_steps: int = 1,
        generator: torch.Generator | None = None,
    ):

        if target_latent_height is None:
            target_latent_height = latents_small.shape[2] * 2

        if target_latent_width is None:
            target_latent_width = latents_small.shape[3] * 2

        schedule = self._schedule(
            num_inference_steps, latents_small.device, latents_small.dtype
        )
        latents = torch.normal(
            mean=0,
            std=1,
            size=(1, 32, target_latent_height, target_latent_width),
            dtype=latents_small.dtype,
            device="cuda",
            generator=generator,
        )
        self.upscaler_unet.eval()

        for t, dt in schedule:
            predicted_noise = self.upscaler_unet(
                sample=latents,
                timestep=t,
                latents_small=latents_small,
            )
            # FlowMatchEulerDiscreteScheduler.step, same operations in the same
            # order (bit-identical): upcast, Euler step, back to the model dtype.
            latents = (latents.to(torch.float32) + dt * predicted_noise).to(
                predicted_noise.dtype
            )

        return latents
