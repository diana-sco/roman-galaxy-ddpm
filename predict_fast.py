""" Generate galaxies with a DDPM model

Jake Lee 2026-04-22 v1
jake.h.lee@jpl.nasa.gov
"""
import sys
import torch
from PIL import Image
import argparse
import h5py
import numpy as np
import time
from data import RomanDDPMPipeline
from diffusers import DPMSolverMultistepScheduler

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Generate images using Roman DDPMPipeline')
    parser.add_argument('--pipeline_path', type=str, required=True, help='Path to the pre-trained pipeline')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size for image generation')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps for diffusion')
    parser.add_argument('--total_images', type=int, default=25, help='Total number of images to generate')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output HDF5 file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., cuda:0, cpu)')

    args = parser.parse_args()
    print(args.device)

    pipeline_path = f"{args.pipeline_path}"
    pipeline = RomanDDPMPipeline.from_pretrained(pipeline_path, low_cpu_mem_usage=True).to(args.device)

    # Switch to DPMSolverMultistepScheduler for faster inference
    # Keep the original DDPMScheduler for training compatibility
    # Set solver_order=3 for unconditional sampling (recommended for DDPM)
    pipeline.median = 0
    pipeline.sigma = 0.02297293
    pipeline.rescale = 0.25
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, 
                                                                 solver_order=3,
                                                                 use_karras_sigmas=True)
    pipeline.scheduler.config.solver_order = 3

    images = []

    random_gen = torch.manual_seed(12)
    for start in range(0, args.total_images, args.batch_size):
        batch_images = pipeline(
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            generator=random_gen,
            output_type='array',
            return_dict=False
        )
        images.extend(batch_images)

        if len(images) >= args.total_images:
            break

    data = np.array(images[:args.total_images])
    data = data.transpose(0,3,1,2)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset("stamps", data=data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{args.output_file} was written in {elapsed_time:.2f} seconds")
