import torch

from fluxrt.flow_upscaler.upscaler_unet import UpscalerUNet
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class FlowUpscalerPipeline:
    def __init__(
        self, upscaler_unet: UpscalerUNet, scheduler: FlowMatchEulerDiscreteScheduler
    ):
        self.upscaler_unet = upscaler_unet
        self.scheduler = scheduler

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

        self.scheduler.set_timesteps(num_inference_steps, mu=1.0)
        latents = torch.normal(
            mean=0,
            std=1,
            size=(1, 32, target_latent_height, target_latent_width),
            dtype=latents_small.dtype,
            device="cuda",
            generator=generator,
        )
        self.upscaler_unet.eval()

        for t in self.scheduler.timesteps:
            latent_model_input = latents
            t = t.to(latents_small.device, latents_small.dtype).view(1)
            predicted_noise = self.upscaler_unet(
                sample=latent_model_input,
                timestep=t,
                latents_small=latents_small,
            )
            latents = self.scheduler.step(predicted_noise, t, latents).prev_sample

        return latents
