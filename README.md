## EDDO: Exoplanet Detection with Differentiable Optics
This repository contains the implementation of the differentiable optics pipeline for exoplanet detection from images taken with the James Webb Space Telescope (JWST) Near Infrared Camera (NIRCam), as described in the paper "Exoplanet Imaging via Differentiable Rendering".

## Example: Real JWST Data - HIP 65426
The file `run_JWST_broadband_OPDandPosition.py` contains the code to run the simulations. 
Example command to process real JWST data will be executed by running:
```
./scripts/run.sh
```

## Example: Simple Simulation
The file `run_JWST.py` contains the code to run the simulations. 
Example command to perform simulation will be executed by running:
```
./scripts/run_JWST_test.sh
```
## Simulation Command Option Explanations
The `--iters` flag sets the number of optimization iterations.

The `--num_t` flag sets the number of measurements (default as 1).

The `--lr` flag sets the optimization learning rate.

The `--scene_name` flag sets the folder name where the results will be saved.

The `--photon_noise` flag controls the amount of photon noise added to the measurements. 
```
    # Example of adding photon noise
    out = ... # noiseless output of the forward model
    random_shot_noise = torch.randn_like(out) * torch.sqrt(out) * args.photon_noise
    out_noisy = out + random_shot_noise
```

The `--contrast_mult` and `--contrast_exp` flags control the contrast of the exoplanet. 
```
    planet_contrast = args.contrast_mult * (10 ** args.contrast_exp)
```

The `--error_ratio` flag controls the deviation of the true wavefront error from the initially assumed wavefront error. 

- For now, the initial wavefront error is a real measurement `OPDs_JWST_dates/opd_2022-08-11T04_06_35.842.npy`, its next real measurement is `OPDs_JWST_dates/opd_2022-08-14T17_58_37.084.npy`, and the error ratio determines the interpolation between these two measurements. 
- The default value is 0.01, which means that the true error underlying the measurement is 1% different from the measured error. Assuming that the two measurements are taken at 100 hours apart, the error ratio can be interpreted as how much time has passed since the initial measurement.
