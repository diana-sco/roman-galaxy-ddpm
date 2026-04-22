""" Data related classes

Jake Lee 2026-04-22 v1
jake.h.lee@jpl.nasa.gov
"""
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from astropy.visualization import make_lupton_rgb

def asinh_scale(data, median=0.000503, sigma=0.006711):
    """
    Apply an asinh scaling to the data.

    This function scales the input data using the arcsinh function, which is useful
    for handling data with a wide dynamic range. The formula used is:
    scaled = sigma * arcsinh((data - median) / (sigma * 3))

    Parameters:
    data (array-like): The input data to be scaled.
    median (float): The median value used in the scaling formula. Default is 0.000503.
    sigma (float): The sigma value used in the scaling formula. Default is 0.006711.
    Returns:
    array-like: The scaled data.
    """
    scaled = sigma * np.arcsinh((data - median) / (sigma * 3))
    return scaled

def inverse_asinh_scale(scaled, median=0.000503, sigma=0.006711):
    """
    Apply the inverse asinh scaling to the data.

    This function scales the input data using the inverse of the arcsinh function.
    The formula used is:
    data = (sinh(scaled / sigma) * sigma * 3) + median

    Parameters:
    scaled (array-like): The input scaled data to be inverse scaled.
    median (float): The median value used in the scaling formula. Default is 0.000503.
    sigma (float): The sigma value used in the scaling formula. Default is 0.006711.
    Returns:
    array-like: The inverse scaled data.
    """
    data = (np.sinh(scaled / sigma) * sigma * 3) + median
    return data

class RomanDataset(Dataset):
    """Pytorch Dataset for Roman HDF5 dataset."""
    def __init__(self, hdf5_path, device='cpu', asinh_median=0.000503, asinh_sigma=0.006711, rescale=0.1):
        self.hdf5_path = hdf5_path
        self.device = device
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.median = asinh_median
        self.sigma = asinh_sigma
        self.rescale = rescale

        with h5py.File(self.hdf5_path, 'r') as f:
            self.data = f['stamps'][:]

        # Filter out NaN images
        self.data = self.data[~np.isnan(self.data).any(axis=(1, 2, 3))]
        # Apply Asinh
        self.data = asinh_scale(self.data, self.median, self.sigma)
        # Rescale to a reasonable min-max range
        self.data = self.data / self.rescale

        self.data = torch.tensor(self.data, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        return image


# Custom Pipeline to remove inline normalization
# From https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/ddpm/pipeline_ddpm.py#L35
class RomanDDPMPipeline(DiffusionPipeline):
    """
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler, asinh_median=0.000503, asinh_sigma=0.006711, rescale=0.1):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.median = asinh_median
        self.sigma = asinh_sigma
        self.rescale = rescale

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample


        # Removed lines
        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = image * self.rescale
        image = inverse_asinh_scale(image, self.median, self.sigma)

        if output_type == "pil":
            raise NotImplementedError("PIL is not supported for RomanDDPMPipeline")

        if not return_dict:
            return image

        return ImagePipelineOutput(images=image)


def apply_make_lupton_rgb(images: np.ndarray) -> np.ndarray:
    """
    Apply the `make_lupton_rgb` function to each RGB image in a batch.

    Parameters:
        images (torch.Tensor): A tensor of shape (b, 3, h, w).

    Returns:
        np.ndarray: An array of processed images.
    """
    b, _, h, w = images.shape
    result_images = []

    for i in range(b):
        # Extract each color channel and convert to numpy array
        red = images[i, :, :, 2]
        green = images[i, :, :, 1]
        blue = images[i, :, :, 0]

        # Apply make_lupton_rgb
        lupton_image = make_lupton_rgb(red, green, blue, stretch=1)
        # This is shape (h, w, 3)

        # Append the result to the list of processed images
        result_images.append(lupton_image)
        # Now (b, h, w, 3)

    return np.array(result_images)
