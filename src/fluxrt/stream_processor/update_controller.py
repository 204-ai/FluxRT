import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as FF

import time


class UpdateController:
    """
    Controls cache updates inside the model through mask values.

    Output mask values:
    - 0: skip, not exectute, not update
    - 1: execute only, no update
    - 2: both execute and update
    """

    def __init__(
        self,
        config: dict,
        height: int,
        width: int,
        compression_ratio: int,
        text_seq_len: int = 512,
        device="cuda",
        dtype=torch.bfloat16,
        reset_period=None,
        reference_image_seq_len: int = None,
    ):
        self.height = height
        self.width = width
        self.compression_ratio = compression_ratio
        self.text_seq_len = text_seq_len
        self.device = device
        self.dtype = dtype
        self.reference_image_seq_len = reference_image_seq_len

        self.mask_height = height // compression_ratio
        self.mask_width = width // compression_ratio

        self.cached_frame = torch.zeros(1, 3, height, width, device=device, dtype=dtype)

        if reset_period is not None:
            self.previous_reset = time.time()

        self.reset_period = reset_period
        self.requires_reset = False
        # One-shot (phase, period): also recompute the patches with
        # (x + y) % period == phase on the next frame (rolling refresh, see
        # ModelInferenceSubprocess._advance_prompt_travel).
        self.refresh = None
        self.text_is_valid = False
        self.reference_image_is_valid = False
        self.config = config

        self.mask_calculation_method = config.get("mask_calculation_method", "auto")
        self.always_update_image_cache = config.get("always_update_image_cache", True)
        # Change detection: mean squared RGB difference (in [-1, 1] units) a patch
        # must exceed to be recomputed, and how many patches the recompute region
        # grows by. Higher threshold / less dilation = fewer tokens, staler edges.
        self.mask_threshold = float(config.get("mask_threshold", 0.1))
        self.mask_dilation = int(config.get("mask_dilation", 2))

        self.requires_update_image_cache = True

        if self.mask_calculation_method == "manual":
            self.mask = None

    def set_mask(self, mask: torch.Tensor):
        """
        Only works when mask_calculation_method is set to "manual"
        Args:
            mask: tensor of shape (1, h // compression_ratio, w // compression_ratio)
        """
        if self.mask_calculation_method != "manual":
            raise ValueError("Mask calculation method is not set to manual.")

        self.mask = mask

    def update_image_cache(self):
        self.requires_update_image_cache = True

    def update_and_get_mask(self, frame: torch.Tensor):
        """
        Args:
            frame: tensor of shape (1, 3, h, w)
        Returns:
            mask: tensor of shape (1, h // compression_ratio, w // compression_ratio)
        """
        if self.mask_calculation_method == "manual":
            if self.mask is None:
                return torch.zeros(
                    1,
                    self.mask_height,
                    self.mask_width,
                    device=self.device,
                    dtype=torch.int32,
                )
            return self.mask

        if self.reset_period is not None:
            if time.time() - self.previous_reset > self.reset_period:
                self.requires_reset = True

        if self.requires_reset:
            self.cached_frame = frame
            self.previous_reset = time.time()
            self.requires_reset = False
            return (
                torch.ones(
                    1,
                    self.mask_height,
                    self.mask_width,
                    device=self.device,
                    dtype=torch.int32,
                )
                * 2
            )

        frame_blurred = F.gaussian_blur(frame, kernel_size=3, sigma=0.5)
        cached_blurred = F.gaussian_blur(self.cached_frame, kernel_size=3, sigma=0.5)

        difference = (cached_blurred - frame_blurred) ** 2
        difference = difference.mean(dim=1, keepdim=True)

        difference_mask = torch.max_pool2d(
            difference, (self.compression_ratio, self.compression_ratio)
        )
        difference_mask_dilated = difference_mask > self.mask_threshold
        for _ in range(self.mask_dilation):
            difference_mask_dilated = (
                FF.max_pool2d(
                    difference_mask_dilated.float(), kernel_size=3, stride=1, padding=1
                )
                > 0
            )
        if self.refresh is not None:
            phase, period = self.refresh
            self.refresh = None
            ys = torch.arange(self.mask_height, device=self.device).view(-1, 1)
            xs = torch.arange(self.mask_width, device=self.device).view(1, -1)
            difference_mask_dilated = difference_mask_dilated | ((xs + ys) % period == phase)

        difference_mask_upsampled = FF.interpolate(
            difference_mask_dilated.float(),
            size=(self.height, self.width),
            mode="nearest",
        )
        self.cached_frame = torch.where(
            difference_mask_upsampled.to(torch.bool).expand(-1, 3, -1, -1),
            frame,
            self.cached_frame,
        )

        image_mask = difference_mask_dilated.squeeze(1).to(torch.int32)

        if self.requires_update_image_cache:
            if not self.always_update_image_cache:
                self.requires_update_image_cache = False
            return image_mask * 2

        return image_mask

    def use_text_mask(self):
        if self.text_is_valid:
            mask = torch.zeros(
                1, self.text_seq_len, device=self.device, dtype=torch.int32
            )
        else:
            mask = (
                torch.ones(1, self.text_seq_len, device=self.device, dtype=torch.int32)
                * 2
            )
            self.text_is_valid = True

        return mask

    def use_reference_image_mask(self):
        if self.reference_image_seq_len is None:
            return None

        if self.reference_image_is_valid:
            mask = torch.zeros(
                1, self.reference_image_seq_len, device=self.device, dtype=torch.int32
            )
        else:
            mask = (
                torch.ones(
                    1,
                    self.reference_image_seq_len,
                    device=self.device,
                    dtype=torch.int32,
                )
                * 2
            ).to(torch.int32)
            self.reference_image_is_valid = True

        return mask

    def reset_cache(self):
        self.requires_reset = True
        self.text_is_valid = False
        self.reference_image_is_valid = False
