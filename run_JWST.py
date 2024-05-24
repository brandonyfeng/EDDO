import os
import tqdm
import imageio
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F

from dl_utils import pixel_coords, crop_to, circle, arcsec2rad, partial_MFT

DEVICE = 'cuda'
ffmpeg_kargs = {'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 4, }


def load_data(N, data_dir, opd_data_dir):
    aperture = np.load(f'{data_dir}/primary_transmission_1024.npy')
    aperture = np.flip(aperture, axis=0) # Flipping to make aperture coincide with Lyot
    lyot = np.load(f'{data_dir}/circlyotstop_transmission_1024.npy')
    fpm = np.load(f'{data_dir}/transmission_MASK335R_4_5um_2048.npy')
    nircam_opd = np.load(f'{data_dir}/FDA_NIRCamLWA_opd_1024.npy')

    #aperture_opd = np.load(f'{data_dir}/primary_EXAMPLE_opd_1024.npy')
    aperture_opd = np.load(f'{opd_data_dir}/opd_2022-08-11T04_06_35.842.npy')
    aperture_opd = np.flip(aperture_opd, axis=0) # Flipping to make aperture coincide with Lyot
    sampledWFEs = np.repeat(np.array([aperture_opd]), N, axis=0)

    aperture_opd2 = np.load(f'{opd_data_dir}/opd_2022-08-14T17_58_37.084.npy')
    aperture_opd2 = np.flip(aperture_opd2, axis=0) # Flipping to make aperture coincide with Lyot
    sampledWFEs_2 = np.repeat(np.array([aperture_opd2]), N, axis=0)


    return aperture, lyot, fpm, nircam_opd, sampledWFEs, sampledWFEs_2


class G_Tensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_imgs):
        super().__init__()
        self.num_imgs = num_imgs
        self.data = nn.Parameter(torch.zeros((num_imgs, x_dim, y_dim, 64)), requires_grad=True)
        self.proj = nn.Linear(64, 64)
        self.proj2 = nn.Linear(64, 1)

    def forward(self):
        out = self.data @ self.proj.weight.t()
        out = torch.sin(out)
        out = out @ self.proj2.weight.t()
        out = out.squeeze(-1)
        return out


class Wavefront(nn.Module):
    def __init__(self, npixels: int, diameter: float, wavelength: float, flux: float, angles = None):
        super().__init__()
        self.wavelength = nn.Parameter(torch.from_numpy(np.asarray(wavelength, float)), requires_grad=False)
        self.pixel_scale = nn.Parameter(torch.from_numpy(np.asarray(diameter / npixels, float)), requires_grad=False)
        self.wavenumber = 2 * np.pi / self.wavelength
        self.npixels = npixels
        self.diameter = diameter
        self.flux = flux
        self.coordinates = nn.Parameter(pixel_coords(self.npixels, self.diameter), requires_grad=False)
        if angles is None:
            angles = torch.zeros(2)
        self.angles = nn.Parameter(angles, requires_grad=True)
        self.reset()

    def reset(self):
        if hasattr(self, 'amplitude'):
            self.amplitude.data = torch.ones_like(self.amplitude.data) / self.npixels**1
            self.phase.data = torch.zeros_like(self.phase.data)
        else:
            self.amplitude = nn.Parameter(torch.ones((1, self.npixels, self.npixels), dtype=torch.float64) / self.npixels**1)
            self.phase = nn.Parameter(torch.zeros((1, self.npixels, self.npixels), dtype=torch.float64))

    def get_phasor(self):
        opd = self.get_tilt_opd()
        return self.amplitude * torch.exp(1j * (self.phase + opd))

    def get_tilt_opd(self):
        opd = -(self.angles[:, None, None] * self.coordinates).sum(0)
        opd = opd[None] * self.wavenumber
        return opd

    def tilt(self, angles):
        """
        Tilts the wavefront by the (x, y) angles.

        Parameters
        ----------
        angles : Array, radians
            The (x, y) angles by which to tilt the wavefront.

        Returns
        -------
        wavefront : Wavefront
            The tilted wavefront.
        """
        coords = self.coordinates
        opd = -(angles[:, None, None] * coords).sum(0)
        opd = opd[None]
        return self.add_opd(opd)

    def add_opd(self, opd):
        self.phase.data = self.phase.data + self.wavenumber * opd

    def propagate(self, phasor = None, pad: int = 2):
        npixels = self.npixels

        if pad > 1:
            _npixels = (npixels * (pad - 1)) // 2
            phasor = torch.nn.functional.pad(phasor, (_npixels, ) * 4)

        phasor = torch.fft.fftshift(torch.fft.ifft2(phasor), dim=[-2, -1])

        return phasor

    def forward(self, phasor, layer):
        new_phasor = phasor * layer
        denom = new_phasor.abs() ** 2
        denom = torch.sum(denom, dim=(1, 2), keepdim=True) ** 0.5
        return new_phasor / denom

    def forward_fpm(self, phasor, layer):
        npixels_in = self.npixels
        phasor = self.propagate(phasor, pad=2)
        new_phasor = phasor * layer
        new_phasor = torch.fft.fft2(torch.fft.fftshift(new_phasor, dim=[-2, -1]))
        new_phasor = crop_to(new_phasor, npixels_in)

        return new_phasor

    def forward_wfe(self, phasor, layer):
        wfe = torch.exp(1j * layer * self.wavenumber)
        new_phasor = phasor * wfe
        return new_phasor


class PointPropagate(nn.Module):
    def __init__(self, aperture, lyot, fpm, nircam_opd, *args):
        super().__init__()
        self.aperture = aperture
        self.lyot = lyot
        self.fpm = fpm
        self.nircam_opd = nircam_opd
        x_mat, y_mat, mult = partial_MFT(*args)
        self.x_mat = nn.Parameter(x_mat, requires_grad=False)
        self.y_mat = nn.Parameter(y_mat, requires_grad=False)
        self.mult = nn.Parameter(torch.tensor(mult, dtype=torch.float64), requires_grad=False)

    def forward(self, wavefront, wfe):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward_wfe(phasor, wfe)
        phasor = wavefront.forward(phasor, self.aperture)
        phasor = wavefront.forward_fpm(phasor, self.fpm)
        phasor = wavefront.forward(phasor, self.lyot)
        phasor = wavefront.forward_wfe(phasor, self.nircam_opd)

        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out

    def forward_val(self, wavefront):
        phasor = wavefront.get_phasor()
        phasor = wavefront.forward(phasor, self.aperture)
        phasor = (self.y_mat.T @ phasor) @ self.x_mat
        phasor *= self.mult
        w = wavefront.flux ** 0.5
        out = (torch.abs(phasor) * w) ** 2

        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='./FilesJWSTMasks/masks_1024', type=str)
    parser.add_argument('--opd_data_dir', default='./OPDs_JWST_dates', type=str)
    parser.add_argument('--scene_name', default='guidestar', type=str)
    parser.add_argument('--num_t', help='Number of measurements', default=32, type=int)
    parser.add_argument('--contrast_mult', default=1, type=float)
    parser.add_argument('--contrast_exp', default=-8, type=int)
    parser.add_argument('--vis_freq', default=100, type=int)
    parser.add_argument('--iters', default=200, type=int)
    parser.add_argument('--lr', default=1e-10, type=float)
    parser.add_argument('--offset_x', default=2.8481e-6, type=float)
    parser.add_argument('--offset_y', default=1.5481e-6, type=float)
    parser.add_argument('--photon_noise', default=0.0, type=float, help='Photon noise level')

    parser.add_argument('--loss_fn', default='L1', type=str)
    parser.add_argument('--error_ratio', default=0.01, type=float)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--log_progress', action='store_true')
    parser.add_argument('--INR', action='store_true')

    args = parser.parse_args()

    # Planet contrast
    planet_contrast = args.contrast_mult * (10 ** args.contrast_exp)
    # Ratio of error to add to the WFE
    error_ratio = args.error_ratio
    # Number of measurements
    N = args.num_t
    # Number of iterations
    iters = args.iters
    # Learning rate
    lr = args.lr
    # Visualization frequency (in iterations)
    vis_freq = args.vis_freq
    # epsilon added to log to avoid log(0)
    vis_norm_eps = 1e-3

    # Wavelength
    wavelen = 4.5e-6
    # Number of pixels in the wavefront
    wf_npix = 1024
    # Diameter of the aperture
    diameter = 6.56
    # Number of pixels in the PSF
    psf_npix = 80
    # Pixel scale in our detector, in arcseconds. For our case, around 14 miliarcseconds per pixel is expected.
    psf_pixel_scale = 0.062424185
    # Radius of the PSF in pixels
    psf_radius = (psf_npix * arcsec2rad(psf_pixel_scale)) / 2
    # Offset of the planet from the center of the PSF
    offset = torch.FloatTensor([args.offset_x, args.offset_y]).view(2)

    # Load data from args.data_dir
    aperture, lyot, fpm, nircam_OPD, sampledWFEs, sampledWFEs_2 = load_data(N=N, data_dir=args.data_dir, opd_data_dir=args.opd_data_dir)

    coords = pixel_coords(wf_npix, diameter)
    aperture = circle(coords, 0.5 * diameter).to(DEVICE)[None]
    nircam_OPD = torch.tensor(nircam_OPD, dtype=torch.float64).to(DEVICE)[None]
    lyot = torch.FloatTensor(lyot).to(DEVICE)[None]
    fpm = torch.FloatTensor(fpm).to(DEVICE)[None]
    sampledWFEs = torch.tensor(sampledWFEs).float()
    sampledWFEs = F.interpolate(sampledWFEs.unsqueeze(1), size=(wf_npix, wf_npix), mode='bilinear').squeeze()

    sampledWFEs_2 = torch.tensor(sampledWFEs_2).float()
    sampledWFEs_2 = F.interpolate(sampledWFEs_2.unsqueeze(1), size=(wf_npix, wf_npix), mode='bilinear').squeeze()

    if len(sampledWFEs.shape) == 2:
        sampledWFEs = sampledWFEs.unsqueeze(0).repeat(2, 1, 1)
    if not args.iid:
        a, b = sampledWFEs[:1], sampledWFEs[1:2]
        sampledWFEs = torch.cat([a * e + b * (1 - e) for e in np.linspace(0, 1, N)])
    sampledWFEs = sampledWFEs.to(DEVICE)
    sampledWFEs_2 = sampledWFEs_2.to(DEVICE)

    ############
    # Set up export directories
    vis_dir = f'{args.root_dir}/vis/{args.scene_name}/N{args.num_t}/{args.contrast_mult}x10_{args.contrast_exp}_err_{error_ratio}_off_{args.offset_x}_{args.offset_y}'
    if args.iid:
        vis_dir += '_iid'
    if args.loss_fn == 'L2':
        vis_dir += '_L2_loss'
    if args.photon_noise > 0:
        vis_dir += f'_pho_noise_{args.photon_noise}'
    os.makedirs(f'{args.root_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    total_photons_real = 11120936 # total number of photons in the science image detector of JWST from Aarynn's program (rough estimate)
    peak_pixel_photons_real = 71095 # number of photons at the brightest pixel at science image detector, using the same estimate as above
    
    # Calibrating star flux
    peak_pixel_intensity_fraction_sim = 0.004 # Empirically, about this much of the original flux ends up in the brightest
                                                # pixel of the simulation.
    flux_fraction_to_detector_star_sim = 0.799 # Empirically, the flux at the detector in this simulation is about 79.9% of the
                                        # flux specified in the flux parameter of the wavefront in the simulation. 
    input_flux_star = 1 / flux_fraction_to_detector_star_sim * total_photons_real # Rough estimate

    # Calibrating planet flux
    desired_contrast_ratio = planet_contrast
    ratio_peak_occulted_to_peak_unocculted = 0.0011 # In webbPSF, the peak pixel of the occulted star PSF is about 0.0011 times the
                                                    # brightness of the peak pixel of the unocculted star PSF
    peak_photons_star_unocculted = peak_pixel_photons_real / ratio_peak_occulted_to_peak_unocculted
    peak_photons_planet_sim = desired_contrast_ratio * peak_photons_star_unocculted

    peak_pixel_to_total_detector_planet_sim = 0.0238 # Empirically, the peak pixel of the planet in the simulations has about 
                                                    # 2.38% of the total brightness in the detector.

    flux_fraction_to_detector_planet_sim = 0.896 # Empirically, the flux at the detector in this simulation is about 89.6% of the
                                        # flux specified in the flux parameter of the wavefront in the simulation when the point source
                                        # is offset by the offset value used by default here [2.8481e-6, 1.5481e-6].

    input_flux_planet = peak_photons_planet_sim / peak_pixel_to_total_detector_planet_sim / flux_fraction_to_detector_planet_sim
    
    flux_1, flux_2 = input_flux_star, input_flux_planet

    # Set up the wavefront objects
    plane_wave_1 = Wavefront(wf_npix, diameter, wavelen, flux_1)
    plane_wave_1 = plane_wave_1.to(DEVICE)
    plane_wave_2 = Wavefront(wf_npix, diameter, wavelen, flux_2, offset)
    plane_wave_2 = plane_wave_2.to(DEVICE)

    # Set up the propagation model parameters
    shift = [0.0, 0.0]
    pixel = True
    focal_length = None
    npixels = wf_npix
    inverse = False
    true_pixel_scale = diameter / npixels
    psf_pixel_scale = arcsec2rad(psf_pixel_scale)
    prop_args = (npixels, wavelen, true_pixel_scale, psf_npix, psf_pixel_scale, focal_length, shift, pixel, inverse)

    # Set up the propagation objects
    p1_model = PointPropagate(aperture, lyot, fpm, nircam_OPD, *prop_args)
    p2_model = PointPropagate(aperture, lyot, fpm, nircam_OPD, *prop_args)
    p1_model = p1_model.to(DEVICE)
    p2_model = p2_model.to(DEVICE)

    # Visualize the resulting PSF
    out_1, out_2 = [], []
    out_1 = p1_model(plane_wave_1, sampledWFEs).detach().cpu()
    out_2 = p2_model(plane_wave_2, sampledWFEs).detach().cpu()
    out = out_1 + out_2

    # Visualize subtracted PSF using the ground truth WFE
    est_residual = out - out_1
    est_residual = est_residual.detach().cpu().numpy()
    est_residual_log10 = np.log10(est_residual + vis_norm_eps)
    est_residual_log10 = (est_residual_log10 - est_residual_log10.min()) / (est_residual_log10.max() - est_residual_log10.min())
    est_residual_log10 = np.uint8(cm.inferno(est_residual_log10) * 255)
    imageio.mimsave(f'{vis_dir}/vis_est_residual_gt.mp4', est_residual_log10, 'FFMPEG', **ffmpeg_kargs)

    # Find planet location in image pixels
    planet_coords = (offset / psf_pixel_scale) + (psf_npix / 2)
    planet_coords = planet_coords.int().cpu().numpy()
    planet_coords = [planet_coords[1], planet_coords[0]]

    # Plot the plantet location on a grid
    image = np.zeros((psf_npix, psf_npix))
    image[planet_coords[0], planet_coords[1]] = 1
    plt.imsave(f'{vis_dir}/planet_location.png', image, cmap='gray')

    # crop the residual PSF around the planet location
    begin_x, end_x = planet_coords[0] - 5, planet_coords[0] + 5
    begin_y, end_y = planet_coords[1] - 5, planet_coords[1] + 5

    if args.photon_noise > 0:
        random_shot_noise = torch.randn_like(out) * torch.sqrt(out) * args.photon_noise
        out_noisy = out + random_shot_noise
        out = torch.clamp(out_noisy, min=0.0)

    observations = out.detach().contiguous().to(DEVICE)
    wfe_gt = sampledWFEs
    wfe_delta = sampledWFEs_2 - sampledWFEs

    out_arr = out.numpy()
    out_arr_log10 = np.log10(out_arr + 1e-10)
    out_arr_log10 = np.array([(a - a.min()) / (a.max() - a.min()) for a in out_arr_log10])
    out_log10 = np.uint8(cm.inferno(out_arr_log10) * 255)
    imageio.mimsave(f'{vis_dir}/vis.mp4', out_log10, 'FFMPEG', **ffmpeg_kargs)

    # Add error to the WFE
    #wfe_off = error_ratio * torch.rand_like(wfe_gt) * wfe_gt.abs()
    wfe_off = error_ratio * wfe_delta
    wfe_batch = sampledWFEs + wfe_off
    wfe_batch = wfe_batch.contiguous()
    wfe_batch.requires_grad = True

    # Visualize the subtracted PSF using the initial WFE with error
    with torch.no_grad():
        e_pred = []
        for batch in range(0, N, 32):
            cur_wfe = wfe_batch[batch:batch + 32]
            pred_1 = p1_model(plane_wave_1, cur_wfe)
            e_pred.append(pred_1.detach().cpu())
        e_pred = torch.cat(e_pred)
    est_residual = torch.relu(out - e_pred)
    est_residual = est_residual.detach().cpu().numpy()
    est_residual_log10 = np.log10(est_residual + vis_norm_eps)
    est_residual_log10 = (est_residual_log10 - est_residual_log10.min()) / (est_residual_log10.max() - est_residual_log10.min())
    est_residual_log10 = np.uint8(cm.inferno(est_residual_log10) * 255)
    imageio.mimsave(f'{vis_dir}/vis_est_res_log10_init.mp4', est_residual_log10, 'FFMPEG', **ffmpeg_kargs)
    imageio.imsave(f'{vis_dir}/vis_est_res_log10_init.png', np.mean(est_residual_log10, axis=0)[..., :3].astype(np.uint8))
    imageio.imsave(f'{vis_dir}/vis_est_res_log10_init_crop.png', np.mean(est_residual_log10, axis=0)[begin_x:end_x, begin_y:end_y, :3].astype(np.uint8))

    imageio.imsave(f'{vis_dir}/vis_measurement.png', np.mean(out_log10, axis=0).astype(np.uint8))

    init_est_star_psf = e_pred.detach().cpu().numpy()
    init_est_residual_psf = est_residual.copy()

    # Record the initial losses
    wfe_err, img_err = F.l1_loss(wfe_batch, wfe_gt), F.l1_loss(e_pred, observations.cpu())
    img_loss_arr, wfe_err_arr = [], []
    img_loss_arr.append(img_err.item())
    wfe_err_arr.append(wfe_err.item())

    # Set up the optimizer and scheduler
    optimizer = torch.optim.AdamW([wfe_batch], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters // 1, eta_min=1e-20)

    if args.INR:
        INR_net = G_Tensor(wfe_batch.shape[-1], wfe_batch.shape[-1], len(wfe_batch))
        INR_net = INR_net.to(DEVICE)
        lr = 1e-10
        optimizer = torch.optim.AdamW(INR_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-20)
        wfe_obs = wfe_batch.detach()

    if args.log_progress:
        progress_arr = []

    # Start the optimization loop
    tbar = tqdm.tqdm(range(iters + 1))
    for i in tbar:
        optimizer.zero_grad()

        if args.INR:
            INR_out = INR_net()
            wfe_batch = wfe_obs + INR_out

        batch_ids = torch.arange(N)
        cur_wfe = wfe_batch[batch_ids]
        pred_1 = p1_model(plane_wave_1, cur_wfe)

        if args.photon_noise > 0:
            random_shot_noise = torch.randn_like(pred_1) * torch.sqrt(pred_1) * args.photon_noise
            pred_1 = pred_1 + random_shot_noise
            pred_1 = torch.clamp(pred_1, min=0.0)

        if args.loss_fn == 'L2':
            loss = (pred_1 - observations[batch_ids]).pow(2).sum() / (199 ** 2 * 32)
        else:
            loss = (pred_1 - observations[batch_ids]).abs().sum() / (199 ** 2 * 32)

        loss.backward()
        optimizer.step()
        scheduler.step()

        e_loss = loss.item() / len(cur_wfe)
        wfe_err = F.l1_loss(wfe_batch, wfe_gt)
        img_loss_arr.append(e_loss)
        wfe_err_arr.append(wfe_err.item())

        if i % vis_freq == 0 and i > 0:
            with torch.no_grad():
                e_pred = p1_model(plane_wave_1, wfe_batch).detach().cpu()
            est_residual = torch.relu(out - e_pred).detach().cpu().numpy()
            est_residual_log10 = np.log10(est_residual + vis_norm_eps)
            est_residual_log10 = (est_residual_log10 - est_residual_log10.min()) / (est_residual_log10.max() - est_residual_log10.min())
            est_residual_log10 = np.uint8(255 * cm.inferno(est_residual_log10))
            imageio.mimsave(f'{vis_dir}/vis_est_res_log10_{i}.mp4', est_residual_log10, 'FFMPEG', **ffmpeg_kargs)
        tbar_out = {'loss': e_loss, 'wfe_err': wfe_err.item()}
        tbar.set_postfix(tbar_out)

        if args.log_progress:
            with torch.no_grad():
                e_pred = p1_model(plane_wave_1, wfe_batch).detach().cpu()
            _est_residual = torch.relu(out - e_pred).detach().cpu().numpy()
            _est_residual_log10 = np.log10(_est_residual + vis_norm_eps)
            _est_residual_log10 = (_est_residual_log10 - _est_residual_log10.min()) / (_est_residual_log10.max() - _est_residual_log10.min())
            progress_arr.append(np.mean(_est_residual_log10, axis=0))
    imageio.imsave(f'{vis_dir}/vis_est_res_log10.png', np.mean(est_residual_log10, axis=0).astype(np.uint8))
    imageio.imsave(f'{vis_dir}/vis_est_res_log10_crop.png', np.mean(est_residual_log10, axis=0)[begin_x:end_x,  begin_y:end_y].astype(np.uint8))

    if args.log_progress:
        progress_out_dir = f'{vis_dir}/progress'
        os.makedirs(progress_out_dir, exist_ok=True)
        for i, img in enumerate(est_residual_log10):
            imageio.imsave(f'{progress_out_dir}/final_{i}.png', img)

        progress_arr = np.stack(progress_arr)
        progress_arr = np.uint8(255 * cm.inferno(progress_arr))
        for i, img in enumerate(progress_arr):
            if i % 5 == 0:
                imageio.imsave(f'{progress_out_dir}/progress_{i}.png', img)
        imageio.mimsave(f'{progress_out_dir}/../progress.mp4', progress_arr[::50], 
                        'FFMPEG', **{'macro_block_size': None, 'ffmpeg_params': ['-s','256x256', '-v', '0'], 'fps': 5, })

    with torch.no_grad():
        est_star_psf = p1_model(plane_wave_1, wfe_batch).detach().cpu().numpy()
    true_star_psf = out_1.numpy()
    est_residual_psf = est_residual
    true_residual_psf = (out - out_1).detach().cpu().numpy()

    psnr_init_star = peak_signal_noise_ratio(true_star_psf / true_star_psf.max(), init_est_star_psf / true_star_psf.max(), data_range=1)
    ssim_init_star = structural_similarity(true_star_psf / true_star_psf.max(), init_est_star_psf / true_star_psf.max(), data_range=1, channel_axis=0)
    psnr_star = peak_signal_noise_ratio(true_star_psf / true_star_psf.max(), est_star_psf / true_star_psf.max(), data_range=1)
    ssim_star = structural_similarity(true_star_psf / true_star_psf.max(), est_star_psf / true_star_psf.max(), data_range=1, channel_axis=0)

    div_norm = true_residual_psf.max()
    true_residual_psf = true_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm
    init_est_residual_psf = init_est_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm
    est_residual_psf = est_residual_psf[:, begin_x:end_x, begin_y:end_y] / div_norm

    psnr_init_residual = peak_signal_noise_ratio(true_residual_psf, init_est_residual_psf, data_range=1)
    ssim_init_residual = structural_similarity(true_residual_psf, init_est_residual_psf, data_range=1, channel_axis=0)
    snr_init_residual = np.mean(init_est_residual_psf) / np.std(init_est_residual_psf)
    print(f'PSNR for the initial residual PSF: {psnr_init_residual:.2f} dB')
    print(f'SSIM for the initial residual PSF: {ssim_init_residual:.4f}')
    print(f'SNR for the initial residual PSF: {snr_init_residual:.2f}')

    psnr_residual = peak_signal_noise_ratio(true_residual_psf, est_residual_psf, data_range=1)
    ssim_residual = structural_similarity(true_residual_psf, est_residual_psf, data_range=1, channel_axis=0)
    snr_residual = np.mean(est_residual_psf) / np.std(est_residual_psf)
    print(f'PSNR for the residual PSF: {psnr_residual:.2f} dB')
    print(f'SSIM for the residual PSF: {ssim_residual:.4f}')
    print(f'SNR for the residual PSF: {snr_residual:.2f}')

    wfe_mse = F.mse_loss(wfe_batch / wfe_gt.max(), wfe_gt / wfe_gt.max()).item()
    wfe_psnr = np.log10(1 / wfe_mse)

    quality_arr = np.array([psnr_init_star, ssim_init_star, psnr_init_residual, ssim_init_residual,
                            psnr_star, ssim_star, psnr_residual, ssim_residual, wfe_mse, wfe_psnr])
    np.save(f'{vis_dir}/quality.npy', quality_arr)

    # Plot the loss curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(img_loss_arr)
    ax[0].set_title('Image Loss')
    ax[0].set_xlabel('Iterations')
    ax[1].plot(wfe_err_arr)
    ax[1].set_title('WFE Loss')
    ax[1].set_xlabel('Iterations')
    plt.savefig(f'{vis_dir}/losses.png')

    # Compare the GT WFEs and final WFEs
    vis_wfe_gt = wfe_gt.detach().cpu().numpy()
    vis_wfe_batch = wfe_batch.detach().cpu().numpy()
    vis_wfe = np.concatenate([vis_wfe_gt, vis_wfe_batch], axis=2)
    vis_wfe = (vis_wfe - vis_wfe.min()) / (vis_wfe.max() - vis_wfe.min())
    vis_wfe = np.uint8(cm.viridis(vis_wfe) * 255)
    ffmpeg_kargs = {'macro_block_size': None, 'ffmpeg_params': ['-s','512x256', '-v', '0'], 'fps': 4}
    imageio.mimsave(f'{vis_dir}/vis_wfe.mp4', vis_wfe, 'FFMPEG', **ffmpeg_kargs)
