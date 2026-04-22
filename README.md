# Diffusion-based Galaxy Simulations for the Roman High Latitude Survey 

Diana Scognamiglio, Jake H. Lee, Eric Huff, Sergi H. Hildebrandt, and Shoubaneh Hemmati,
_Accepted_ 2026.

This repository contains the code necessary to reproduce the results presented in the above publication. The train, test, and generated datasets, along with trained model weights, and a packaged version of this software, is provided on Zenodo: https://doi.org/10.5281/zenodo.19699521

## Dependencies

- pytorch: https://pytorch.org/get-started/locally/
- huggingface diffusers: https://huggingface.co/docs/diffusers/en/index
- astropy: https://docs.astropy.org/en/stable/
- W&B: https://docs.wandb.ai/models

## Model Training

The model training script uses Weights & Biases for experiment tracking.

The following call was used to train the model as presented:

```
python train.py \
    --train-dataset data/train_2p0cut.hdf5 \
    --test-dataset data/test_2p0cut.hdf5 \
    --outdir ./ \
    --timesteps 500 \
    --crop 56 \
    --channels 128 \
    --batch 16 \
    --epochs 100 \
    --lr 0.0001 \
    --lr-warmup 500 \
    --device cuda:0 \
    --wandb-entity 'your-wandb-entity' \
    --wandb-project 'roman-ddpm' \
    --wandb-name 'conv-2p0cut'
```

## Model Inference

The following call was used to generate roman-like galaxies with the trained model:

```
python predict_fast.py \
    --pipeline_path model/ \
    --batch_size 32 \
    --num_inference_steps 25 \
    --total_images 10000 \
    --output_file generated_2p0cut.hdf5 \
    --device cuda:0
```

As described in Section 5.1 of the paper, these generated galaxies were further filtered by enforcing that the center of the galaxy lie within a radius of $r_{\rm cut} = 20$ pixels from the stamp center, a minimum size $R_{\rm m} \geq 2.0$ pixels, and a minimum Kron signal-to-noise ratio $SNR \geq 3$.

### Acknowledgements
The research was carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004), © 2026. All rights reserved. In particular, this work was funded through the Jet Propulsion Laboratory's Spontaneous Concept Research and Technology Development program, which supported this research. The High Performance Computing resources used in this work were provided by funding from the JPL Enterprise Technology, Strategy, and Cybersecurity Directorate. The authors also acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing computational resources that have contributed to the research results reported within this paper. Portions of this work were completed at Duke University. D.S. thanks Michael Troxel and Arun Kannawadi for insightful comments that improved the manuscript. This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes at the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-03127 for JWST. These observations are associated with program #3215 and #1963.