""" Train a DDPM model for galaxy generation

Jake Lee 2026-04-22 v1
jake.h.lee@jpl.nasa.gov
"""
import os
import argparse
from datetime import datetime
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import  MSELoss
from torchvision.transforms import CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, Compose

from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

import wandb

from data import RomanDataset, RomanDDPMPipeline
from data import asinh_scale


os.environ["WANDB__SERVICE_WAIT"] = "300"
SIGMA = 0.02297293
SCALE = 0.25

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion model for Roman galaxy generation")

    # Add arguments based on DefaultConfig class attributes
    parser.add_argument('--train-dataset', type=str, default='/scratch/euclid-gan/2025-roman/datasets/v1_train.hdf5',
                       help='Path to the train dataset file')
    parser.add_argument('--test-dataset', type=str, default='/scratch/euclid-gan/2025-roman/datasets/v1_test.hdf5',
                       help='Path to the test dataset file')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory')

    parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps')

    parser.add_argument('--crop', type=int, default=56, help='Crop size')
    parser.add_argument('--channels', type=int, default=128, help='Initial channel')


    parser.add_argument('--batch', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--attn', action='store_true', help='Enable attn layers')
    parser.add_argument('--cos', action='store_true', help='Enable cosine scheduling')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-warmup', type=int, default=500, help='Learning rate warmup steps')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')

    parser.add_argument('--wandb-entity', type=str, default='roman-diffusion', help='WandB entity.')
    parser.add_argument('--wandb-project', type=str, default='v1', help="WandB project to be logged to.")
    parser.add_argument('--wandb-name', type=str, default='', help="Project name to be appended to timestamp for wandb name.")

    args = parser.parse_args()

    # Dataset definition
    train_dataset = RomanDataset(args.train_dataset, \
                                    args.device, \
                                    asinh_median = 0, \
                                    asinh_sigma = SIGMA, \
                                    rescale = SCALE)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    test_dataset = RomanDataset(args.test_dataset, \
                                args.device, \
                                asinh_median = 0, \
                                asinh_sigma = SIGMA, \
                                rescale = SCALE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # Model definition

    ch = args.channels
    model = UNet2DModel(
        sample_size=args.crop,         # Input image size, 56x56
        in_channels=3,          # YJH
        out_channels=3,         # YJH
        layers_per_block=2,     # 1: small, 2: large
        center_input_sample=False,
        block_out_channels=(ch, ch*2, ch*4),  # Channel sizes per block, typically doubles each layer
        down_block_types = (
            "DownBlock2D",
            f"{'Attn' if args.attn else ''}DownBlock2D",
            f"{'Attn' if args.attn else ''}DownBlock2D",
        ),
        up_block_types = (
            #"AttnUpBlock2D",    # Could be replaced with UpBlock2D, relatively slow
            f"{'Attn' if args.attn else ''}UpBlock2D",
            f"{'Attn' if args.attn else ''}UpBlock2D",
            "UpBlock2D"
        ),
        dropout=0.05
    )
    model = model.to(args.device)

    # Noise scheduler
    # Options: https://huggingface.co/docs/diffusers/en/api/schedulers/ddpm#diffusers.DDPMScheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.timesteps,
        beta_schedule='squaredcos_cap_v2' if args.cos else 'linear')

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    # Ref: https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup,
        num_training_steps=args.epochs * len(train_dataloader)
    )

    # Set up WandB tracking
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_{args.wandb_name}")
    run = wandb.init(
        project = args.wandb_project,
        entity = args.wandb_entity,
        name = timestamp,
        dir = './',
        config = vars(args)
    )

    # Set up loss function
    criterion = MSELoss()

    # Augmentation
    train_transforms = [
        CenterCrop(args.crop),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5)
    ]
    train_transforms = Compose(train_transforms)

    val_transforms = CenterCrop(args.crop)

    # Training loop
    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            batch = train_transforms(batch)

            optimizer.zero_grad()

            # Sample noise to add to the stamps
            noise = torch.randn_like(batch).to(batch.device)
            bs = batch.size(0)

            # Sample a random timestep for each stamp in the batch
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=batch.device).long()

            # Add noise to the stamps according to the noise magnitude
            noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

            # Forward pass
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Compute loss
            loss = criterion(noise_pred, noise)

            train_loss += loss.cpu().item()

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # log
            run.log({"loss_train_iter": loss.cpu().item()})

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = val_transforms(batch)

                noise = torch.randn_like(batch).to(batch.device)
                bs = batch.size(0)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=batch.device).long()
                noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = criterion(noise_pred, noise)
                val_loss += loss.cpu().item()

        val_loss /= len(test_dataloader)

        # Per-epoch visualization
        pipeline = RomanDDPMPipeline(unet=model, scheduler=noise_scheduler)

        images = pipeline(batch_size=5, 
                            generator=torch.manual_seed(42),
                            num_inference_steps=args.timesteps,
                            output_type='array',
                            return_dict=False)

        # images is shape (5, h, w, c)
        scaled_images = []
        for image in images:
            image = asinh_scale(image)
            image = np.clip(image, -SIGMA, SIGMA*4)
            image = (image + SIGMA) / (SIGMA*5)
            image = (image*255).astype(np.uint8)
            image = np.flip(image, axis=0)
            image = Image.fromarray(image, mode='RGB')
            scaled_images.append(image)

        run.log({
            "loss_train_epoch": train_loss,
            "loss_val_epoch": val_loss,
            "example_gen": [wandb.Image(image) for image in scaled_images],
            "epoch": epoch})

        # After training, save pipeline
        if (epoch+1) % 5 == 0:
            epoch_outdir = os.path.join(args.outdir, timestamp, f"ep{epoch}")
            os.makedirs(epoch_outdir, exist_ok=True)
            pipeline.save_pretrained(epoch_outdir)

if __name__ == "__main__":
    main()
